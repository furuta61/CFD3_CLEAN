import io, hashlib
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import pytesseract

# ---- iPhone縦4銘柄 固定ROI（必要なら微調整可）----
ROI_SPEC = {
    "JP225":  (0.33, 0.18, 0.67, 0.25),
    "NAS100": (0.33, 0.30, 0.67, 0.37),
    "GER40":  (0.33, 0.42, 0.67, 0.49),
    "XAUUSD": (0.33, 0.54, 0.67, 0.61),
}


def _crop(img: np.ndarray, xyxy_rel: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = int(w * xyxy_rel[0]); y1 = int(h * xyxy_rel[1])
    x2 = int(w * xyxy_rel[2]); y2 = int(h * xyxy_rel[3])
    return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]


def _pre_gray(src: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if src.ndim == 3 else src
    g = cv2.convertScaleAbs(g, alpha=1.15, beta=8)
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return b


def _tess_digits(img_roi: np.ndarray) -> Optional[float]:
    b = _pre_gray(img_roi)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=1)
    if b.shape[1] < 600:
        b = cv2.resize(b, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789."
    txt = pytesseract.image_to_string(b, config=cfg).strip().replace(",", "")
    while ".." in txt:
        txt = txt.replace("..", ".")
    try:
        v = float(txt)
        return v if 0.01 <= v <= 1_000_000 else None
    except:
        return None


# -------- Paddle フォールバック（互換初期化）--------
_paddle = None
def _lazy_paddle():
    """PaddleOCR をバージョン差異を吸収して安全初期化"""
    global _paddle
    if _paddle is not None:
        return _paddle
    from paddleocr import PaddleOCR
    # ① できれば簡略高速設定
    try:
        _paddle = PaddleOCR(
            lang="en",
            use_textline_orientation=False,  # 新系
            use_doc_preprocessor=False,      # 新系
            show_log=False                   # 新系
        )
        return _paddle
    except Exception:
        pass
    # ② 旧系の引数（angle_cls）ならそれで
    try:
        _paddle = PaddleOCR(lang="en", use_angle_cls=False)
        return _paddle
    except Exception:
        pass
    # ③ 最小構成（確実に通る）
    _paddle = PaddleOCR(lang="en")
    return _paddle


def _paddle_number(img_roi: np.ndarray) -> Optional[float]:
    ocr = _lazy_paddle()
    res = ocr.ocr(img_roi)  # 余計なkwargsは付けない（バージョン差異対策）
    texts = []

    # v5系: dict形式 / 旧系: list形式 をどちらも吸収
    if isinstance(res, list) and res and isinstance(res[0], dict) and "rec_texts" in res[0]:
        for page in res:
            texts.extend(page.get("rec_texts", []))
    else:
        # 旧形式: [ [ [box, (txt,score)], ... ] ]
        for page in res if isinstance(res, list) else []:
            for item in page:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    _box, ts = item
                    if isinstance(ts, (list, tuple)) and ts:
                        texts.append(str(ts[0]))

    nums = []
    for t in texts:
        t = str(t or "").strip().replace(",", "")
        if not t:
            continue
        # 数字と小数点のみ
        if any(c for c in t if (not c.isdigit() and c != ".")):
            continue
        try:
            v = float(t)
            if 0.01 <= v <= 1_000_000:
                nums.append(v)
        except:
            pass
    if not nums:
        return None
    nums_sorted = sorted(nums)
    import numpy as np
    median = float(np.median(nums_sorted))
    return min(nums_sorted, key=lambda v: abs(v - median))


# -------- メイン + キャッシュ --------
_cache: Dict[str, Dict[str, Optional[float]]] = {}


def extract_watchlist_mobile_v12_turbo(path_or_bytes) -> Dict[str, Optional[float]]:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        raw = bytes(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            raw = f.read()
    key = hashlib.sha1(raw).hexdigest()
    if key in _cache:
        out = dict(_cache[key])
        out["engine"] = "mobile-v12-turbo(cache)"
        return out

    img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))[:, :, ::-1]
    out: Dict[str, Optional[float]] = {"JP225": None, "NAS100": None, "GER40": None, "XAUUSD": None}

    for sym, rel in ROI_SPEC.items():
        roi = _crop(img, rel)
        v = _tess_digits(roi)
        if v is None:
            v = _paddle_number(roi)
        out[sym] = v

    out["engine"] = "mobile-v12-turbo"
    _cache[key] = dict(out)
    return out
