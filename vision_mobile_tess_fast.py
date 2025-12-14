import re, cv2, numpy as np from typing import Dict, Tuple, Optional, List
import pytesseract

# 1320x2868 を基準にしたスマホ4銘柄ROI（横幅余裕・縦は広め）
BASE_ROI = {
    "JP225":  (430,  520, 890, 320),
    "NAS100": (430,  870, 890, 320),
    "GER40":  (430, 1210, 890, 320),
    "XAUUSD": (430, 1550, 890, 320),
}

# 妥当レンジ（ノイズ除去用）
RANGE = {
    "JP225":  (10_000, 80_000),
    "NAS100": ( 8_000, 50_000),
    "GER40":  ( 8_000, 50_000),
    "XAUUSD": ( 1_000,  5_000),
}

_num = re.compile(r"\d+(?:\.\d+)?")

def _rescale(xywh: Tuple[int,int,int,int], W: int, H: int) -> Tuple[int,int,int,int]:
    x,y,w,h = xywh
    X1 = int(x * W / 1320.0)
    Y1 = int(y * H / 2868.0)
    X2 = int((x+w) * W / 1320.0)
    Y2 = int((y+h) * H / 2868.0)
    X1 = max(0, min(X1, W-1)); X2 = max(0, min(X2, W))
    Y1 = max(0, min(Y1, H-1)); Y2 = max(0, min(Y2, H))
    return X1, Y1, max(1, X2-X1), max(1, Y2-Y1)

def _prep_variants(gray: np.ndarray) -> List[np.ndarray]:
    outs = []
    # コントラスト強調
    g1 = cv2.convertScaleAbs(gray, alpha=1.6, beta=10); outs.append(g1)
    # 反転 + 固定二値
    inv = cv2.bitwise_not(g1)
    for th in (170, 160, 150):
        outs.append(cv2.threshold(inv, th, 255, cv2.THRESH_BINARY)[1])
    # 適応二値（局所）
    outs.append(cv2.adaptiveThreshold(
        cv2.bitwise_not(gray), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, 8
    ))
    return outs

def _pick_best(nums: List[float], name: str) -> Optional[float]:
    if not nums: return None
    lo, hi = RANGE.get(name, (0, float("inf")))
    inrng = [v for v in nums if lo <= v <= hi]
    if inrng:
        # 小数あり長桁優先
        inrng.sort(key=lambda v: (len(str(int(v))), v))
        return inrng[-1]
    nums.sort(key=lambda v: (len(str(int(v))), v))
    return nums[-1]

def _tess_digits(img: np.ndarray) -> List[float]:
    # 画像は前処理済みの二値or高コントラストを期待
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789."
    text = pytesseract.image_to_string(img, lang="eng", config=cfg) or ""
    vals = []
    for m in _num.findall(text):
        try:
            vals.append(float(m))
        except:
            pass
    return vals

def _scan_one(roi_bgr: np.ndarray, name: str) -> Optional[float]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # 縦ずれに強くする：上下微スライド + 高さ微拡張
    H, W = gray.shape
    offsets = [-80, -40, 0, 40, 80]
    heights = [1.00, 1.25, 1.50]
    candidates: List[float] = []
    for dy in offsets:
        for mh in heights:
            y0 = max(0, min(H-1, int(H*0.0 + dy)))
            h2 = int(min(H - y0, H * mh))
            crop = gray[y0:y0+h2, :]
            for p in _prep_variants(crop):
                nums = _tess_digits(p)
                candidates.extend(nums)
    return _pick_best(candidates, name)

def extract_watchlist_mobile_tess(path: str) -> Dict[str, Optional[float]]:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("画像を読み込めませんでした。パスを確認してください。")
    H, W = img.shape[:2]
    out: Dict[str, Optional[float]] = {}
    for name, rect in BASE_ROI.items():
        x,y,w,h = _rescale(rect, W, H)
        val = _scan_one(img[y:y+h, x:x+w], name)
        out[name] = val
    out["engine"] = "mobile-vTESS"
    return out
