# vision_fixedroi.py — fixedroi-mobile-v10（スマホスクショ専用）
import io
import re
from typing import Dict, Any
import numpy as np
from PIL import Image
import pytesseract

# ---- ROI（あなたのiPhoneスクショに最適化された座標） ----
ROI_MAP = {
    "JP225":  (300, 350, 1100, 500),
    "NAS100": (300, 700, 1100, 850),
    "GER40":  (300, 1030, 1100, 1180),
    "XAUUSD": (300, 1370, 1100, 1520),
}

# 小数点が必ず1つ、かつ小数部1〜2桁／整数部は最大5桁（= 99999.x まで）

NUM_PATTERN = re.compile(r"[0-9]+\.[0-9]+|[0-9]+")

def extract_number(text: str) -> float | None:
    m = NUM_PATTERN.search(text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group())
    except:
        return None

def ocr_roi(img: Image.Image, roi: tuple) -> float | None:
    x1, y1, x2, y2 = roi
    crop = img.crop((x1, y1, x2, y2))

    # OCR 最適化:白黒反転・二値化・拡大
    gray = crop.convert("L")
    gray = gray.resize((gray.width * 2, gray.height * 2))

    text = pytesseract.image_to_string(gray, lang="eng", config="--psm 7")
    return extract_number(text)

def extract_watchlist_fixed_v10(image_bytes: bytes) -> Dict[str, Any]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = {}

    for symbol, roi in ROI_MAP.items():
        val = ocr_roi(img, roi)
        results[symbol] = val

    results["engine"] = "fixedroi-mobile-v10"
    return results
