# vision_local.py  (local-ocr-v2.1 robust)
import io
import re
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image
import pytesseract
import cv2

# 数値トークン（:,/ を含むものは除外）
_NUM_TOKEN = re.compile(r"^[0-9]+(?:[.,][0-9]+)*$")

# ★ 価格の上限閾値（ボリューム等の巨大値を除外）
#   JP225/NAS100/GER40/GOLD/WTI/FX を想定して 100,000 を上限に設定
MAX_PRICE = 100_000.0


def _to_float(tok: str) -> Optional[float]:
    t = tok.strip().replace(",", "")
    # 時刻/日付の記号があれば除外
    if ":" in t or "/" in t:
        return None
    try:
        v = float(t)
        if not (0.01 <= v <= MAX_PRICE):
            return None
        return v
    except Exception:
        return None


def _to_gray(img_pil: Image.Image) -> np.ndarray:
    if img_pil.mode != "L":
        img_pil = img_pil.convert("L")
    arr = np.array(img_pil)
    return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr


def _preprocess(img_pil: Image.Image) -> np.ndarray:
    gray = _to_gray(img_pil)
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = cv2.resize(bin_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return bin_img  # 1ch


def _ocr_items(bin_gray: np.ndarray) -> List[Dict[str, Any]]:
    cfg = "--psm 6 --oem 3"
    data = pytesseract.image_to_data(bin_gray, lang="eng+jpn", config=cfg, output_type=pytesseract.Output.DICT)
    items = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf_raw = data["conf"][i]
        try:
            conf = int(conf_raw)
        except Exception:
            conf = -1
        if text == "" or conf < 50:  # ★ 低信頼をより強く除外
            continue
        items.append({
            "text": text,
            "conf": conf,
            "x": data["left"][i],
            "y": data["top"][i],
            "w": data["width"][i],
            "h": data["height"][i],
            "block": data["block_num"][i],
            "line": data["line_num"][i],
        })
    return items


def _same_row(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    ay, ah = a["y"], a["h"]
    by, bh = b["y"], b["h"]
    if a["line"] == b["line"]:
        return True
    return abs((ay + ah / 2) - (by + bh / 2)) <= max(ah, bh)


def _nearest_number_to_right(items: List[Dict[str, Any]], label: Dict[str, Any]) -> Optional[float]:
    lx, lw = label["x"], label["w"]
    cands = []
    for it in items:
        if it["x"] <= lx + lw:  # 右側のみ
            continue
        if not _same_row(label, it):  # 同一行のみ
            continue
        if not _NUM_TOKEN.match(it["text"]):
            continue
        val = _to_float(it["text"])
        if val is None:
            continue
        dx = it["x"] - (lx + lw)
        cands.append((dx, -it["conf"], val))  # 右に近く・高信頼ほど優先
    if not cands:
        return None
    cands.sort()
    return cands[0][2]


_LABELS = {
    "price": [r"現在値", r"現値", r"約定値", r"\bPRICE\b", r"\bLAST\b"],
    "high": [r"高値", r"\bHIGH\b", r"高\b"],
    "low": [r"安値", r"\bLOW\b", r"安\b"],
    "bid": [r"\bBID\b", r"売\s*気\s*配", r"売り", r"売\b"],
    "ask": [r"\bASK\b", r"買\s*気\s*配", r"買い", r"買\b"],
}


def _find_by_label(items: List[Dict[str, Any]], pats: List[str]) -> Optional[Dict[str, Any]]:
    for it in items:
        t = it["text"]
        for p in pats:
            if re.search(p, t, flags=re.IGNORECASE):
                return it
    return None


def _robust_filter(nums: List[float]) -> List[float]:
    """IQRで外れ値除去しつつ、ゼロ/極端値を除外"""
    arr = np.array([v for v in nums if v is not None and v > 0 and v <= MAX_PRICE], dtype=float)
    if arr.size == 0:
        return []
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filt = arr[(arr >= max(0.01, lo)) & (arr <= min(MAX_PRICE, hi))]
    if filt.size == 0:
        return arr.tolist()
    return filt.tolist()


def extract_prices_from_image_local(image_bytes: bytes) -> Dict[str, Any]:
    """
    戻り値: {"bid":float|None,"ask":float|None,"price":float|None,"high":float|None,"low":float|None,"engine":"local-ocr-v2"}
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        bin_img = _preprocess(img)
        items = _ocr_items(bin_img)

        out = {"bid": None, "ask": None, "price": None, "high": None, "low": None}

        # 1) ラベル近傍優先抽出
        for key in ["price", "high", "low", "bid", "ask"]:
            lab = _find_by_label(items, _LABELS[key])
            if lab:
                val = _nearest_number_to_right(items, lab)
                if val is not None:
                    out[key] = val

        # 2) 全数値→ロバスト抽出（IQR）
        raw_nums = []
        for it in items:
            if _NUM_TOKEN.match(it["text"]):
                v = _to_float(it["text"])
                if v is not None:
                    raw_nums.append(v)
        nums = _robust_filter(raw_nums)

        # 3) 価格中心ウィンドウで再選別（±50%以内に限定）
        def _window(center: float, seq: List[float]) -> List[float]:
            if center is None or not seq:
                return seq
            lo, hi = center * 0.5, center * 1.5
            return [v for v in seq if lo <= v <= hi]

        # price 推定
        if out["price"] is None and nums:
            med = float(np.median(nums))
            out["price"] = min(nums, key=lambda v: abs(v - med))

        # high/low 推定（price から外れた巨大値を除く）
        wnums = _window(out["price"], nums)
        if wnums:
            if out["high"] is None:
                out["high"] = max(wnums)
            if out["low"] is None:
                out["low"] = min(wnums)

        # 4) 整合性・Bid/Ask 補完
        p = out["price"]
        if out["high"] is not None and out["low"] is not None and out["high"] < out["low"]:
            out["high"], out["low"] = out["low"], out["high"]

        # Bid/Ask が price から大きく外れていれば捨てて補完
        def _reasonable(v: Optional[float], center: Optional[float]) -> bool:
            return v is not None and center is not None and (center * 0.5 <= v <= center * 1.5)

        if not _reasonable(out["bid"], p):
            out["bid"] = p - 0.5 if p is not None else None
        if not _reasonable(out["ask"], p):
            out["ask"] = p + 0.5 if p is not None else None

        # 下限ゼロ回避
        for k in ["price", "high", "low", "bid", "ask"]:
            if isinstance(out.get(k), float) and out[k] is not None and out[k] < 0.01:
                out[k] = None

        out["engine"] = "local-ocr-v2"
        if out["price"] is None:
            raise RuntimeError("価格（現在値）の抽出に失敗。画面の倍率/鮮明度を上げて再撮影してください。")
        return out
    except Exception as e:
        raise RuntimeError(f"[LocalOCR解析エラー] {e}")


# ---- デバッグ可視化（任意）----
def dump_debug_ocr(image_bytes: bytes, out_path: str = "debug_ocr.png") -> str:
    img = Image.open(io.BytesIO(image_bytes))
    bin_img = _preprocess(img)
    items = _ocr_items(bin_img)

    # 1ch → BGRにして矩形描画
    vis = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    for it in items:
        x, y, w, h = it["x"], it["y"], it["w"], it["h"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(vis, it["text"], (x, max(0, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)
    return out_path
