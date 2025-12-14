# vision_watchlist.py  — watchlist(1枚)→4銘柄一括抽出 v2 (JP UI最適化)
import io
import re
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image
import pytesseract
import cv2

# ---- 銘柄名の日本語/UI表記を広めにカバー ----
NAME_PATTERNS: Dict[str, List[str]] = {
    "JP225": [r"日本\s*225", r"日本225", r"日経\s*225", r"日経225", r"\bJP\s*225\b", r"\bJP225\b", r"Japan\s*225"],
    "NAS100": [r"米国\s*NQ\s*100\s*ミニ", r"米国NQ100ミニ", r"米国NQ100", r"NQ\s*100", r"NAS\s*100", r"\bUS\s*100\b", r"ナスダック\s*100", r"NASDAQ\s*100"],
    "GER40": [r"ドイツ\s*40", r"ドイツ40", r"\bGER\s*40\b", r"\bDAX\b", r"Germany\s*40"],
    "XAUUSD": [r"金\s*スポット", r"金スポット", r"\bXAU\s*/?\s*USD\b", r"\bXAUUSD\b", r"\bGOLD\b", r"Gold"],
}

MAX_PRICE = 200_000.0  # 桁外れ除外の上限

_NUM = re.compile(r"^[0-9]+(?:[.,][0-9]+)*$")


def _to_gray(pil: Image.Image) -> np.ndarray:
    if pil.mode != "L":
        pil = pil.convert("L")
    arr = np.array(pil)
    return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr


def _preprocess_both(pil: Image.Image) -> List[np.ndarray]:
    """明暗2通りの前処理画像を返す（暗色UI対策）"""
    g = _to_gray(pil)
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
    except Exception:
        g = cv2.equalizeHist(g)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, bin1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin2 = cv2.bitwise_not(bin1)  # 反転版
    bin1 = cv2.resize(bin1, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    bin2 = cv2.resize(bin2, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return [bin1, bin2]


def _ocr_data(img: np.ndarray, cfg: str) -> List[dict]:
    d = pytesseract.image_to_data(img, lang="eng+jpn", config=cfg, output_type=pytesseract.Output.DICT)
    out = []
    for i in range(len(d["text"])):
        txt = (d["text"][i] or "").strip()
        try:
            conf = int(d["conf"][i])
        except Exception:
            conf = -1
        if not txt or conf < 50:
            continue
        out.append({
            "text": txt,
            "conf": conf,
            "x": d["left"][i],
            "y": d["top"][i],
            "w": d["width"][i],
            "h": d["height"][i],
            "line": d["line_num"][i],
            "block": d["block_num"][i],
        })
    return out


def _merge_items(items: List[dict]) -> List[dict]:
    # 同じテキスト・近接ボックスを単純マージ（重複除去）
    out: List[dict] = []
    for it in items:
        dup = False
        for j, ex in enumerate(out):
            if it["text"] == ex["text"] and abs(it["x"] - ex["x"]) < 6 and abs(it["y"] - ex["y"]) < 6:
                # 大きい方/高confを残す
                if it["conf"] > ex["conf"] or it["h"] > ex["h"]:
                    out[j] = it
                dup = True
                break
        if not dup:
            out.append(it)
    return out


def _run_ocr_dual(pil: Image.Image) -> List[dict]:
    imgs = _preprocess_both(pil)
    all_items: List[dict] = []
    # 一般テキスト
    for im in imgs:
        all_items += _ocr_data(im, "--psm 6 --oem 3")
    # 数値専用（桁取り強化）
    for im in imgs:
        all_items += _ocr_data(im, "--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789.,:OHLCoHLC")
    return _merge_items(all_items)


def _is_number_token(t: str) -> bool:
    return bool(_NUM.match(t)) and (":" not in t and "/" not in t)


def _to_float(t: str) -> Optional[float]:
    s = t.replace(",", "")
    try:
        v = float(s)
        if 0.01 <= v <= MAX_PRICE:
            return v
    except Exception:
        pass
    return None


def _map_name_to_code(name_text: str) -> Optional[str]:
    for code, pats in NAME_PATTERNS.items():
        for p in pats:
            if re.search(p, name_text, flags=re.IGNORECASE):
                return code
    return None


def _pick_price_candidates(items: List[dict]) -> List[dict]:
    # フォント高さと幅でスコアリングし、上位を候補化
    nums = [it for it in items if _is_number_token(it["text"])]
    if not nums:
        return []
    hs = np.array([it["h"] for it in nums], dtype=float)
    ws = np.array([it["w"] for it in nums], dtype=float)
    h_thr = float(np.percentile(hs, 75))  # 上位25%を候補
    w_thr = float(np.percentile(ws, 60))
    cands = []
    for it in nums:
        if it["h"] >= h_thr or it["w"] >= w_thr:
            v = _to_float(it["text"])
            if v is not None:
                cands.append(dict(it, value=v))
    # 縦位置でソート
    cands.sort(key=lambda z: z["y"])
    return cands


def _row_band(y: int, h: int) -> Tuple[int, int]:
    r = int(h * 3.0)
    return max(0, y - r), y + r


def _nearest_name_left(items: List[dict], price_it: dict) -> Optional[str]:
    px = price_it["x"]
    py = price_it["y"]
    ph = price_it["h"]
    y0, y1 = _row_band(py, ph)
    best_code, best_dx = None, None
    for it in items:
        if it["x"] + it["w"] > px:  # 左側のみ
            continue
        cy = it["y"] + it["h"] // 2
        if not (y0 <= cy <= y1):  # 同じ行帯
            continue
        code = _map_name_to_code(it["text"])
        if not code:
            continue
        dx = px - (it["x"] + it["w"])
        if best_dx is None or dx < best_dx:
            best_code, best_dx = code, dx
    return best_code


def _pick_hl_in_band(items: List[dict], y0: int, y1: int) -> Dict[str, Optional[float]]:
    out = {"high": None, "low": None}
    for i, it in enumerate(items):
        cy = it["y"] + it["h"] // 2
        if not (y0 <= cy <= y1):
            continue
        t = it["text"]
        # H: / L: の直後の数値を拾う（'H', 'H:' など許容）
        if t.upper().startswith("H"):
            for j in range(i + 1, min(i + 5, len(items))):
                tv = items[j]["text"]
                if _is_number_token(tv):
                    v = _to_float(tv)
                    if v is not None:
                        out["high"] = v
                        break
        elif t.upper().startswith("L"):
            for j in range(i + 1, min(i + 5, len(items))):
                tv = items[j]["text"]
                if _is_number_token(tv):
                    v = _to_float(tv)
                    if v is not None:
                        out["low"] = v
                        break
    return out


def extract_watchlist_from_image(image_bytes: bytes) -> Dict[str, Dict[str, Any]]:
    """
    1枚のウォッチリスト画像から {code: {price, high?, low?, bid, ask, engine}} を返す
    """
    pil = Image.open(io.BytesIO(image_bytes))
    items = _run_ocr_dual(pil)
    prices = _pick_price_candidates(items)
    result: Dict[str, Dict[str, Any]] = {}

    # 近い縦座標の重複価格を間引く（同一行で複数拾った場合）
    used_y = []

    def _close_to_used(y: int) -> bool:
        return any(abs(y - uy) < 30 for uy in used_y)

    for pit in prices:
        if _close_to_used(pit["y"]):
            continue
        y0, y1 = _row_band(pit["y"], pit["h"])
        code = _nearest_name_left(items, pit)
        if not code:
            continue
        used_y.append(pit["y"])
        hl = _pick_hl_in_band(items, y0, y1)

        result[code] = {
            "price": pit["value"],
            "high": hl.get("high"),
            "low": hl.get("low"),
            "bid": None,
            "ask": None,  # ウォッチ画面には無いので後段で補完
            "engine": "watch-ocr-v2",
        }
    return result


# ---- 任意: デバッグ可視化 ----
def dump_debug_watchlist(image_bytes: bytes, out_path: str = "debug_watchlist.png") -> str:
    pil = Image.open(io.BytesIO(image_bytes))
    imgs = _preprocess_both(pil)
    vis = cv2.cvtColor(imgs[0], cv2.COLOR_GRAY2BGR)
    items = _run_ocr_dual(pil)
    for it in items:
        x, y, w, h = it["x"], it["y"], it["w"], it["h"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(vis, it["text"], (x, max(0, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)
    return out_path
