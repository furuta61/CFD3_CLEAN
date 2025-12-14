# watchlist_fast_calibrated.py — 4銘柄/10秒内・校正保存型OCR
# deps: pillow, numpy, opencv-python-headless, pytesseract
import io, os, json, re, concurrent.futures as fut
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import cv2
from PIL import Image
import pytesseract

CALIB_PATH = "watchlist_calib.json"   # レイアウト別のROI(0-1の割合座標)を保存
LAST_PATH  = "last_quotes.json"       # 前回値バンドの参照(任意)

# 現実レンジ(緩め) + tick
PRICE_RANGE = {
    "JP225":  (20000.0, 70000.0),
    "NAS100": (10000.0, 40000.0),
    "GER40":  (14000.0, 20000.0),
    "XAUUSD": (1500.0,  3000.0),
}
TICK = {"JP225":5.0, "NAS100":0.25, "GER40":1.0, "XAUUSD":0.1}
ORDER = ["JP225","NAS100","GER40","XAUUSD"]
NUM_DEC = re.compile(r"^[0-9]{1,5}(?:,[0-9]{3})*(?:\.[0-9]{1,2})$")

# --------------- 基本ユーティリティ ---------------
def _pil_to_gray_u8(pil: Image.Image) -> np.ndarray:
    if pil.mode != "L": pil = pil.convert("L")
    arr = np.array(pil)
    return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr

def _binarize(gray: np.ndarray) -> List[np.ndarray]:
    g = cv2.GaussianBlur(gray, (3,3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [b, cv2.bitwise_not(b)]

def _layout(gray: np.ndarray) -> str:
    H, W = gray.shape[:2]
    return "pc" if (W/H) >= 1.2 else "mobile"

def _to_val(s: str) -> Optional[float]:
    t = (s or "").strip().replace(" ", "").replace("\n", "")
    if not NUM_DEC.match(t): return None
    try: return float(t.replace(",", ""))
    except: return None

def _in_range(code:str, v:float, last:Optional[float]) -> bool:
    lo, hi = PRICE_RANGE.get(code, (0.01, 1e9))
    if last is not None:
        # 前回値±1.5%の帯を優先(はみ出す場合も最終的には許容)
        band_lo = last * 0.985
        band_hi = last * 1.015
        lo = max(lo, band_lo*0.9)
        hi = min(hi, band_hi*1.1)
    return (lo <= v <= hi)

def _clip(W:int,H:int, box:Tuple[float,float,float,float]) -> Tuple[int,int,int,int]:
    x1,y1,x2,y2 = box
    x1 = max(0, min(W-1, int(x1))); x2 = max(0, min(W-1, int(x2)))
    y1 = max(0, min(H-1, int(y1))); y2 = max(0, min(H-1, int(y2)))
    if x2<=x1: x2=min(W-1,x1+1)
    if y2<=y1: y2=min(H-1,y1+1)
    return x1,y1,x2,y2

# --------------- キャリブ ---------------
_DEFAULT_CALIB = {
    "mobile": {
        # 0-1 の割合座標(あなたのスマホ縦4行前提)
        "JP225":  [0.10, 0.20, 0.97, 0.33],
        "NAS100": [0.10, 0.38, 0.97, 0.51],
        "GER40":  [0.10, 0.56, 0.97, 0.69],
        "XAUUSD": [0.10, 0.74, 0.97, 0.87],
    },
    "pc": {
        # PC横4(あなたの環境に合わせて広め)
        "JP225":  [0.15, 0.36, 0.35, 0.48],
        "NAS100": [0.37, 0.36, 0.57, 0.48],
        "GER40":  [0.59, 0.36, 0.79, 0.48],
        "XAUUSD": [0.81, 0.36, 1.00, 0.48],
    }
}

def _load_calib() -> Dict[str, Dict[str, List[float]]]:
    if os.path.exists(CALIB_PATH):
        try:
            return json.load(open(CALIB_PATH, "r"))
        except Exception:
            pass
    return _DEFAULT_CALIB

def save_calib(calib: Dict[str, Dict[str, List[float]]]) -> None:
    json.dump(calib, open(CALIB_PATH, "w"), indent=2, ensure_ascii=False)

# --------------- OCR ---------------
def _ocr_crop(crop: np.ndarray) -> Optional[float]:
    # PSM 7→13、二値化の正/反転をすべて試す(高速)
    for psm in (7, 13):
        txt = pytesseract.image_to_string(
            crop, lang="eng+jpn",
            config=f"--psm {psm} --oem 1 -c tessedit_char_whitelist=0123456789.,"
        )
        v = _to_val(txt)
        if v is not None:
            return v
    return None

def _prep_crop(gray: np.ndarray, W:int, H:int, frac_box: List[float], scale: float) -> List[np.ndarray]:
    x1 = frac_box[0]*W; y1 = frac_box[1]*H; x2 = frac_box[2]*W; y2 = frac_box[3]*H
    x1,y1,x2,y2 = _clip(W,H,(x1,y1,x2,y2))
    bimgs = _binarize(gray[y1:y2, x1:x2])
    out = []
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    for b in bimgs:
        if b.size==0: continue
        c = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, k, iterations=1)
        out.append(c)
    return out

def _task_one(gray: np.ndarray, layout: str, code: str, frac_box: List[float],
              last: Optional[float], scale: float) -> Tuple[str, Optional[float], np.ndarray]:
    H,W = gray.shape[:2]
    for crop in _prep_crop(gray, W, H, frac_box, scale):
        v = _ocr_crop(crop)
        if v is not None and _in_range(code, v, last):
            return code, v, crop
    return code, None, np.zeros((1,1), dtype=np.uint8)

def _mini_offsets() -> List[Tuple[float,float]]:
    # ちょいズレ救済(±6%×3×3)
    ds = [-0.06, 0.0, 0.06]
    return [(dx,dy) for dy in ds for dx in ds]

def _apply_offset(box: List[float], dx:float, dy:float) -> List[float]:
    x1,y1,x2,y2 = box
    w = x2-x1; h = y2-y1
    return [x1+dx*w, y1+dy*h, x2+dx*w, y2+dy*h]

def _get_last(code: str) -> Optional[float]:
    try:
        d = json.load(open(LAST_PATH, "r"))
        v = d.get(code) or d.get(code.lower())
        if isinstance(v, dict): v = v.get("price")
        return float(v) if v is not None else None
    except Exception:
        return None

# --------------- 公開API ---------------
def extract_watchlist(img_bytes: bytes) -> Dict[str, Dict[str, Any]]:
    pil = Image.open(io.BytesIO(img_bytes))
    gray = _pil_to_gray_u8(pil); H,W = gray.shape[:2]
    layout = _layout(gray)
    calib = _load_calib()[layout]
    scale = 2.6 if layout=="mobile" else 2.2

    # まず固定ROIで4並列
    futures = []
    out: Dict[str, Dict[str, Any]] = {}
    with fut.ThreadPoolExecutor(max_workers=4) as ex:
        for code in ORDER:
            last = _get_last(code)
            futures.append(ex.submit(_task_one, gray, layout, code, calib[code], last, scale))
        for f in futures:
            code, v, crop = f.result()
            if v is not None:
                tick = TICK.get(code, 0.1)
                out[code] = {"price": v, "bid": max(v-tick, 0.01), "ask": v+tick,
                             "engine": f"calib-{layout}-fast"}

    # 欠けはミニ探索(±6%×3×3)だけ実施
    if len(out) < 4:
        for code in ORDER:
            if code in out: continue
            last = _get_last(code)
            base = calib[code]
            got = None
            for dx,dy in _mini_offsets():
                box2 = _apply_offset(base, dx, dy)
                _, v, _ = _task_one(gray, layout, code, box2, last, scale)
                if v is not None:
                    tick = TICK.get(code, 0.1)
                    out[code] = {"price": v, "bid": max(v-tick, 0.01), "ask": v+tick,
                                 "engine": f"calib-{layout}-mini"}
                    got = True
                    break
            if not got:
                # まだダメなら「前回値±1.5%」のバンドを無視してレンジのみ許容
                for dx,dy in _mini_offsets():
                    box2 = _apply_offset(base, dx, dy)
                    H,W = gray.shape[:2]
                    for crop in _prep_crop(gray, W, H, box2, scale):
                        v = _ocr_crop(crop)
                        if v is not None and _in_range(code, v, None):
                            tick = TICK.get(code, 0.1)
                            out[code] = {"price": v, "bid": max(v-tick, 0.01), "ask": v+tick,
                                         "engine": f"calib-{layout}-mini2"}
                            got = True
                            break
                    if got: break
    return out

def debug_overlay(img_bytes: bytes, save_dir="debug_watchlist"):
    os.makedirs(save_dir, exist_ok=True)
    pil = Image.open(io.BytesIO(img_bytes))
    gray = _pil_to_gray_u8(pil); H,W = gray.shape[:2]
    layout = _layout(gray)
    calib = _load_calib()[layout]
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    colors = {"JP225":(255,0,0),"NAS100":(0,255,0),"GER40":(0,255,255),"XAUUSD":(0,128,255)}
    for code in ORDER:
        fx1,fy1,fx2,fy2 = calib[code]
        x1,y1,x2,y2 = _clip(W,H,(fx1*W,fy1*H,fx2*W,fy2*H))
        cv2.rectangle(vis,(x1,y1),(x2,y2),colors[code],3)
        cv2.putText(vis, code, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[code], 2, cv2.LINE_AA)

        # 現在のcropも保存
        bimgs = _binarize(gray[y1:y2, x1:x2])
        for idx, b in enumerate(bimgs):
            path = os.path.join(save_dir, f"{code}_crop_{idx}.png")
            cv2.imwrite(path, b)
    cv2.imwrite(os.path.join(save_dir, f"overlay_{layout}.png"), vis)
    return f"Saved to {save_dir}/"
