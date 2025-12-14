# vision_fixedroi_fast.py — v7 高速×堅牢 固定ウォッチリストOCR（4銘柄）
# 依存: pillow, numpy, opencv-python-headless, pytesseract
import io, re, concurrent.futures as fut, os
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import cv2
from PIL import Image
import pytesseract

# --- 価格レンジ（GMO想定・外れは採用しない） ---
PRICE_RANGE = {
    "JP225":  (20000.0, 70000.0),
    "NAS100": (10000.0, 40000.0),
    "GER40":  (14000.0, 20000.0),   # 16〜18k帯想定
    "XAUUSD": (1500.0,  3000.0),    # 2000台中心
}
TICK = {"JP225":5.0, "NAS100":0.25, "GER40":1.0, "XAUUSD":0.1}

# 小数点必須・整数最大5桁・小数1〜2桁
NUM_DEC = re.compile(r"^[0-9]{1,5}(?:,[0-9]{3})*(?:\.[0-9]{1,2})$")

_last_debug = {
    "gray": None, "layout": None,
    "boxes_primary": None, "boxes_alt": None,
    "wins_used": {}, "wins_used_alt": {},
    "crops": {}, "crops_alt": {},
    "vals": {}, "vals_alt": {}
}

def _pil_to_gray_u8(pil: Image.Image) -> np.ndarray:
    if pil.mode != "L":
        pil = pil.convert("L")
    arr = np.array(pil)
    return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr

def _pre_bin(gray: np.ndarray) -> np.ndarray:
    # 軽量・安定の二値化
    g = cv2.GaussianBlur(gray, (3,3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b

def _layout(gray: np.ndarray) -> str:
    H, W = gray.shape[:2]
    return "pc" if (W/H) >= 1.2 else "mobile"

# --- ROI（あなたの実スクショ向け・primary/alt の2系統） ---
def _boxes_mobile_primary(W:int,H:int)->List[Tuple[int,int,int,int]]:
    x0 = int(W*0.06); x1 = int(W*0.94)
    top = int(H*0.17); bot = int(H*0.90)
    row_h = (bot - top)//4
    boxes=[]
    for i in range(4):
        ry0 = top + i*row_h; ry1 = top + (i+1)*row_h
        px1 = int(x0 + (x1-x0)*0.10); px2 = int(x1 - (x1-x0)*0.03)
        py1 = int(ry0 + (ry1-ry0)*0.10); py2 = int(ry0 + (ry1-ry0)*0.46)
        boxes.append((px1,py1,px2,py2))
    return boxes

def _boxes_mobile_alt(W:int,H:int)->List[Tuple[int,int,int,int]]:
    # 代替: 縦位置をやや下げ・高さ広め（数字が少し下寄りの端末向け）
    x0 = int(W*0.06); x1 = int(W*0.94)
    top = int(H*0.17); bot = int(H*0.90)
    row_h = (bot - top)//4
    boxes=[]
    for i in range(4):
        ry0 = top + i*row_h; ry1 = top + (i+1)*row_h
        px1 = int(x0 + (x1-x0)*0.12); px2 = int(x1 - (x1-x0)*0.02)
        py1 = int(ry0 + (ry1-ry0)*0.18); py2 = int(ry0 + (ry1-ry0)*0.56)
        boxes.append((px1,py1,px2,py2))
    return boxes

def _boxes_pc_primary(W:int,H:int)->List[Tuple[int,int,int,int]]:
    x0 = int(W*0.14); x1 = int(W*0.95)
    y0 = int(H*0.34); y1 = int(H*0.86)
    col_w = (x1 - x0)//4
    boxes=[]
    for i in range(4):
        cx0 = x0 + i*col_w; cx1 = x0 + (i+1)*col_w
        px1 = int(cx0 + (cx1-cx0)*0.10); px2 = int(cx0 + (cx1-cx0)*0.90)
        py1 = int(y0 + (y1-y0)*0.12);   py2 = int(y0 + (y1-y0)*0.36)
        boxes.append((px1,py1,px2,py2))
    return boxes

def _boxes_pc_alt(W:int,H:int)->List[Tuple[int,int,int,int]]:
    # 代替: 横幅を少し広め・縦位置を少し上げる
    x0 = int(W*0.13); x1 = int(W*0.96)
    y0 = int(H*0.33); y1 = int(H*0.85)
    col_w = (x1 - x0)//4
    boxes=[]
    for i in range(4):
        cx0 = x0 + i*col_w; cx1 = x0 + (i+1)*col_w
        px1 = int(cx0 + (cx1-cx0)*0.08); px2 = int(cx0 + (cx1-cx0)*0.92)
        py1 = int(y0 + (y1-y0)*0.10);   py2 = int(y0 + (y1-y0)*0.38)
        boxes.append((px1,py1,px2,py2))
    return boxes

def _clip(W:int,H:int,x1:int,y1:int,x2:int,y2:int)->Tuple[int,int,int,int]:
    x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
    if x2<=x1: x2=min(W-1,x1+1)
    if y2<=y1: y2=min(H-1,y1+1)
    return x1,y1,x2,y2

def _resize_morph(bin_img: np.ndarray, box: Tuple[int,int,int,int], scale: float) -> np.ndarray:
    x1,y1,x2,y2 = box
    crop = bin_img[y1:y2, x1:x2]
    if crop.size == 0: 
        return crop
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, k, iterations=1)
    return crop

def _to_val(txt: str) -> Optional[float]:
    s = (txt or "").strip().replace(" ", "").replace("\n", "")
    if not NUM_DEC.match(s): return None
    try: return float(s.replace(",", ""))
    except: return None

def _ocr_text(crop: np.ndarray, psm:int) -> Optional[float]:
    cfg = f"--psm {psm} --oem 1 -c tessedit_char_whitelist=0123456789.,"
    txt = pytesseract.image_to_string(crop, lang="eng+jpn", config=cfg)
    return _to_val(txt)

def _in_range(code:str, v:float)->bool:
    lo, hi = PRICE_RANGE.get(code, (0.01, 1e9))
    return (lo <= v <= hi)

def _ocr_region_fast(bin_img: np.ndarray, box: Tuple[int,int,int,int],
                     code: str, scale: float=2.4) -> Tuple[Optional[float], np.ndarray]:
    crop = _resize_morph(bin_img, box, scale)
    if crop.size == 0: return None, crop
    for psm in (7, 13):
        v = _ocr_text(crop, psm)
        if v is not None and _in_range(code, v):
            return v, crop
    return None, crop

def _mini_search(bin_img: np.ndarray, base: Tuple[int,int,int,int],
                 code: str, scale_list: List[float], W:int, H:int) -> Tuple[Optional[float], Optional[np.ndarray], Tuple[int,int,int,int]]:
    # 欠け銘柄だけ ±8% × 3x3 × 3スケール × 2PSM
    bx1,by1,bx2,by2 = base
    bw, bh = bx2-bx1, by2-by1
    dx = int(bw * 0.08)
    dy = int(bh * 0.08)
    best_v, best_crop, best_box = None, None, base
    for oy in (-dy, 0, dy):
        for ox in (-dx, 0, dx):
            x1,y1,x2,y2 = _clip(W,H, bx1+ox, by1+oy, bx2+ox, by2+oy)
            for sc in scale_list:
                crop = _resize_morph(bin_img, (x1,y1,x2,y2), sc)
                if crop.size == 0: 
                    continue
                for psm in (7,13):
                    v = _ocr_text(crop, psm)
                    if v is not None and _in_range(code, v):
                        return v, crop, (x1,y1,x2,y2)
    return best_v, best_crop, best_box

def _one_task(bin_img: np.ndarray, box: Tuple[int,int,int,int], code:str, scale:float) -> Tuple[str, Optional[float], np.ndarray]:
    v, crop = _ocr_region_fast(bin_img, box, code, scale)
    return (code, v, crop)

def _try_boxes(gray: np.ndarray, bin_img: np.ndarray, boxes: List[Tuple[int,int,int,int]],
               layout: str, order: List[str], mini_scales: List[float]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Tuple[int,int,int,int]], Dict[str, np.ndarray]]:
    W,H = gray.shape[1], gray.shape[0]
    scale = 2.4 if layout=="pc" else 2.8
    out: Dict[str, Dict[str, Any]] = {}
    wins: Dict[str, Tuple[int,int,int,int]] = {}
    crops: Dict[str, np.ndarray] = {}

    # まず固定ROIで4並列
    with fut.ProcessPoolExecutor(max_workers=4) as ex:
        tasks = [ex.submit(_one_task, bin_img, b, c, scale) for c,b in zip(order, boxes)]
        fut.wait(tasks, timeout=6)

    vals: Dict[str, Optional[float]] = {}
    for c, b, t in zip(order, boxes, tasks):
        v, crop = None, None
        if t.done() and not t.cancelled():
            try:
                c2, v, crop = t.result()
            except Exception:
                v, crop = None, None
        vals[c] = v
        if crop is not None:
            crops[c] = crop
        if v is not None:
            wins[c] = b
            tick = TICK.get(c, 0.1)
            out[c] = {"price": v, "bid": max(v - tick, 0.01), "ask": v + tick,
                      "engine": f"fixedroi-{layout}-v7"}
    # 欠けはミニ探索
    for c, b in zip(order, boxes):
        if vals[c] is None:
            v, crop, box2 = _mini_search(bin_img, b, c, mini_scales, W, H)
            if v is not None:
                crops[c] = crop
                wins[c] = box2
                tick = TICK.get(c, 0.1)
                out[c] = {"price": v, "bid": max(v - tick, 0.01), "ask": v + tick,
                          "engine": f"fixedroi-{layout}-v7-mini"}
    return out, wins, crops

def extract_watchlist_fast(image_bytes: bytes) -> Dict[str, Dict[str, Any]]:
    """
    4銘柄の price/bid/ask を10秒以内で安定抽出。
    手順: 固定ROI→欠けはミニ探索→ まだ欠けるなら代替ROIで同手順。
    """
    pil = Image.open(io.BytesIO(image_bytes))
    gray = _pil_to_gray_u8(pil)
    bin_img = _pre_bin(gray)
    layout = _layout(gray)
    W,H = gray.shape[1], gray.shape[0]
    order = ["JP225","NAS100","GER40","XAUUSD"]
    primary = _boxes_pc_primary(W,H) if layout=="pc" else _boxes_mobile_primary(W,H)
    alt     = _boxes_pc_alt(W,H)     if layout=="pc" else _boxes_mobile_alt(W,H)
    mini_scales = [2.0, 2.6, 3.2] if layout=="pc" else [2.4, 3.0, 3.6]

    out1, wins1, crops1 = _try_boxes(gray, bin_img, primary, layout, order, mini_scales)
    if len(out1) == 4:
        _last_debug.update({"gray":gray,"layout":layout,"boxes_primary":primary,"boxes_alt":alt,
                            "wins_used":wins1,"wins_used_alt":{}, "crops":crops1,"crops_alt":{},
                            "vals":{k:v["price"] for k,v in out1.items()}, "vals_alt":{}})
        return out1

    # 欠けがある→ alt で未取得のみ狙い撃ち
    missing = [c for c in order if c not in out1]
    out2, wins2, crops2 = {}, {}, {}
    if missing:
        # alt側のボックス順は order に対応済みなので、その銘柄の枠だけ抽出
        subset_boxes = [alt[i] for i,_c in enumerate(order)]
        # 未取得分だけ再試行（実装簡易化のため全銘柄altを通し、既取得は上書きしない）
        out_alt, wins_alt, crops_alt = _try_boxes(gray, bin_img, subset_boxes, layout, order, mini_scales)
        for c in missing:
            if c in out_alt:
                out2[c] = out_alt[c]
                wins2[c] = wins_alt.get(c, None)
                crops2[c] = crops_alt.get(c, None)

    out = dict(out1)
    out.update(out2)
    _last_debug.update({"gray":gray,"layout":layout,"boxes_primary":primary,"boxes_alt":alt,
                        "wins_used":wins1,"wins_used_alt":wins2, "crops":crops1,"crops_alt":crops2,
                        "vals":{k:v["price"] for k,v in out1.items()},
                        "vals_alt":{k:v["price"] for k,v in out2.items()}})
    return out

# --- デバッグ書き出し（オーバーレイ＋各Crop） ---
def debug_save_all(out_dir="debug_fixedroi"):
    os.makedirs(out_dir, exist_ok=True)
    if _last_debug["gray"] is None:
        return "No debug context"
    gray = _last_debug["gray"]
    layout = _last_debug["layout"]
    base_p = _last_debug["boxes_primary"]
    base_a = _last_debug["boxes_alt"]
    wins1  = _last_debug["wins_used"]
    wins2  = _last_debug["wins_used_alt"]
    crops1 = _last_debug["crops"]
    crops2 = _last_debug["crops_alt"]
    vals1  = _last_debug["vals"]
    vals2  = _last_debug["vals_alt"]

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    order = ["JP225","NAS100","GER40","XAUUSD"]
    col = {"JP225":(255,0,0),"NAS100":(0,255,0),"GER40":(0,255,255),"XAUUSD":(0,128,255)}

    # 基準枠（primary:細線、alt:点線）
    for box in base_p:
        x1,y1,x2,y2 = box
        cv2.rectangle(vis,(x1,y1),(x2,y2),(160,160,160),1)
    for box in base_a:
        x1,y1,x2,y2 = box
        cv2.rectangle(vis,(x1,y1),(x2,y2),(120,120,120),1, lineType=cv2.LINE_AA)

    # 採用枠（primary→実線太線、alt→太点線）
    for c in order:
        if c in wins1:
            x1,y1,x2,y2 = wins1[c]
            cv2.rectangle(vis,(x1,y1),(x2,y2),col[c],3)
            v = vals1.get(c,"")
            cv2.putText(vis, f"{c}:{v}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col[c], 2, cv2.LINE_AA)
        if c in wins2:
            x1,y1,x2,y2 = wins2[c]
            cv2.rectangle(vis,(x1,y1),(x2,y2),col[c],3, lineType=cv2.LINE_AA)
            v = vals2.get(c,"")
            cv2.putText(vis, f"{c}(alt):{v}", (x1, min(vis.shape[0]-6,y2+18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col[c], 2, cv2.LINE_AA)

    overlay_path = os.path.join(out_dir, f"overlay_{layout}.png")
    cv2.imwrite(overlay_path, vis)

    # 各Crop保存
    for name, crop in {**{f"{k}_p":v for k,v in crops1.items()},
                       **{f"{k}_a":v for k,v in crops2.items()}}.items():
        if crop is not None and crop.size>0:
            cv2.imwrite(os.path.join(out_dir, f"{name}.png"), crop)

    return f"Saved debug to {out_dir}/"
