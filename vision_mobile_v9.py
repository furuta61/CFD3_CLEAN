# vision_mobile_v9.py
# ----------------------------------------
# GMOウォッチリスト特化：銘柄アンカー方式（v9）
# 4銘柄 = JP225 / NAS100 / GER40 / XAUUSD
# 現在値（右側の最大数字1つ）だけを抽出
# ----------------------------------------

import io
import re
import numpy as np
from typing import Dict, Any, Optional, List
from PIL import Image
import pytesseract
import cv2


# ---------- 銘柄 → 正規表現マッピング ----------
ANCHOR_PATTERNS = {
    "JP225":   [r"日.?本.?225", r"JP.?225"],
    "NAS100":  [r"米.?国.?N.?Q.?100", r"NAS.?100"],
    "GER40":   [r"ド.?イ.?ツ.?40", r"GER.?40"],
    "XAUUSD":  [r"金.?ス.?ポ.?ッ.?ト", r"XAU.?USD"],
}

# 現在値として妥当なレンジ
PRICE_RANGE = {
    "JP225":  (20000, 60000),
    "NAS100": (7000, 40000),
    "GER40":  (5000, 40000),
    "XAUUSD": (1000, 3000)
}


# ---------- OCR（全画面 → 単語リスト） ----------
def _ocr_full(image_bytes: bytes) -> List[Dict[str, Any]]:
    img = Image.open(io.BytesIO(image_bytes))

    # Grayscale OCR 安定化
    gray = np.array(img.convert("L"))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    data = pytesseract.image_to_data(
        gray, lang="eng+jpn",
        config="--psm 6 --oem 3",
        output_type=pytesseract.Output.DICT
    )

    items = []
    n = len(data["text"])
    for i in range(n):
        t = (data["text"][i] or "").strip()
        if t == "":
            continue

        try:
            conf = int(data["conf"][i])
        except:
            conf = -1

        if conf < 30:
            continue

        items.append({
            "text": t,
            "conf": conf,
            "x": data["left"][i],
            "y": data["top"][i],
            "w": data["width"][i],
            "h": data["height"][i],
            "line": data["line_num"][i]
        })
    return items


# ---------- 銘柄名を OCR から探してアンカー行を取得 ----------
def _find_anchor_y(items: List[Dict[str, Any]], patterns: List[str]) -> Optional[int]:
    for it in items:
        for p in patterns:
            if re.search(p, it["text"], flags=re.IGNORECASE):
                center_y = it["y"] + it["h"] // 2
                return center_y
    return None


# ---------- アンカー行の右側から最大数値を取得 ----------
def _extract_price_from_line(items: List[Dict[str, Any]], anchor_y: int,
                             tolerance: int = 35) -> Optional[float]:

    row_words = []
    for it in items:
        cy = it["y"] + it["h"] // 2
        if abs(cy - anchor_y) <= tolerance:
            row_words.append(it)

    numbers = []
    for it in row_words:
        token = it["text"].replace(",", "")
        try:
            v = float(token)
            numbers.append(v)
        except:
            continue

    if not numbers:
        return None

    # 右端ほど大きい数字が現在値であることが多い → 最大値
    return max(numbers)


# ---------- 大本命：v9 抽出器 ----------
def extract_watchlist_v9(image_bytes: bytes) -> Dict[str, Any]:
    items = _ocr_full(image_bytes)

    results = {}
    for symbol, patterns in ANCHOR_PATTERNS.items():

        # 1) 銘柄アンカーを取得
        y = _find_anchor_y(items, patterns)
        if y is None:
            results[symbol] = None
            continue

        # 2) アンカー行から現在値らしき最大値を取得
        val = _extract_price_from_line(items, y)
        if val is None:
            results[symbol] = None
            continue

        # 3) 正しいレンジに収まっているか判定
        lo, hi = PRICE_RANGE[symbol]
        if not (lo <= val <= hi):
            results[symbol] = None
            continue

        results[symbol] = val

    results["engine"] = "fixedroi-mobile-v9"
    return results
