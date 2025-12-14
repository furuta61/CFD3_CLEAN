# vision_mobile_v13.py
# iPhone GMOスクショ専用: ラベルを読んで右隣の"太字価格"だけを抽出
# 依存: pip install easyocr pillow opencv-python-headless pytesseract

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
from functools import lru_cache
import re, math, os

import numpy as np
from PIL import Image, ImageOps
import cv2
import easyocr
import pytesseract

# ---- ラベル候補（日本語/英語・ゆらぎ対応） ----
LABELS: Dict[str, List[str]] = {
    "JP225":  ["日本225", "日経", "JP225", "日経225", "にほん225"],
    "NAS100": ["米国NQ100", "米国NQ100ミニ", "NQ100", "ナス", "ナスダック"],
    "GER40":  ["ドイツ40", "GER40", "独40", "ドイツ４０"],
    "XAUUSD": ["金スポット", "XAUUSD", "ゴールド", "金"],
}

# ---- 価格の妥当レンジ（外れ値除去用） ----
RANGE: Dict[str, Tuple[float, float]] = {
    "JP225":  (30000, 80000),
    "NAS100": (10000, 40000),
    "GER40":  (10000, 40000),
    "XAUUSD": (3000, 6000),
}

DIGIT_RE = re.compile(r"(?<![A-Za-z])([0-9]{2,6}(?:[.,][0-9]{1,3})?)")

# ---------- OCR Readers ----------
@lru_cache(maxsize=1)
def _easyocr_reader():
    # 言語は英+日で十分。GPU不要。
    return easyocr.Reader(['ja','en'], gpu=False, verbose=False)

def _tesseract_number(img: np.ndarray) -> Optional[str]:
    # 数字専用の高速読み取り（太字価格向け）
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789.,"
    txt = pytesseract.image_to_string(img, config=cfg)
    txt = txt.strip()
    return txt or None

def _clean_num(s: str) -> Optional[float]:
    s = s.replace(',', '').replace(' ', '')
    s = s.replace('O', '0').replace('o', '0').replace('S', '5')
    m = DIGIT_RE.search(s)
    if not m:
        return None
    try:
        val = float(m.group(1).replace(',', ''))
        return val
    except:
        return None

def _valid(sym: str, v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    lo, hi = RANGE[sym]
    return v if (lo <= v <= hi) else None

# ---------- 画像前処理 ----------
def _load_image(path_or_pil) -> Image.Image:
    img = path_or_pil if isinstance(path_or_pil, Image.Image) else Image.open(path_or_pil)
    img = img.convert('RGB')
    # 解像度不足対策（2倍拡大）
    w, h = img.size
    if min(w, h) < 1400:
        img = img.resize((w*2, h*2), Image.LANCZOS)
    return img

def _to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def _mono_variants(img: Image.Image) -> List[np.ndarray]:
    cv = _to_cv(img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    # 暗背景で白文字 → 反転も試す
    th1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 31, 11)
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 9)
    inv1 = cv2.bitwise_not(th1)
    inv2 = cv2.bitwise_not(th2)
    return [th1, th2, inv1, inv2]

# ---------- ラベル検出 → 右隣ROI生成 ----------
def _find_label_boxes(img: Image.Image) -> Dict[str, Tuple[int,int,int,int]]:
    reader = _easyocr_reader()
    arr = np.array(img)
    H, W = arr.shape[:2]
    results = reader.readtext(arr, detail=1, paragraph=False)

    # 文字列→正規化（空白除去・小文字）
    def norm(s: str) -> str:
        s = re.sub(r"\s+", "", s)
        return s.lower()

    boxes: Dict[str, Tuple[int,int,int,int]] = {}
    for (bbox, text, conf) in results:
        try:
            xs = [int(p[0]) for p in bbox]; ys = [int(p[1]) for p in bbox]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        except:
            continue

        t = norm(text)
        for sym, cand in LABELS.items():
            if sym in boxes:
                continue
            if any(norm(c) in t for c in cand):
                # ラベルの右隣にでかい価格が来るので、右へ幅広にROIを取る
                h = y2 - y1
                roi_x1 = max(0, x2 + int(0.02*W))
                roi_y1 = max(0, y1 - int(0.6*h))
                roi_x2 = min(W, roi_x1 + int(0.70*W))
                roi_y2 = min(H, y2 + int(1.2*h))
                boxes[sym] = (roi_x1, roi_y1, roi_x2, roi_y2)
                break

    # 見つからなかった行は"縦4分割の保険ROI"を使う
    if len(boxes) < 4:
        rowH = H // 4
        fallback = {
            "JP225":  (int(0.30*W), int(0.00*rowH), int(0.95*W), int(0.90*rowH)),
            "NAS100": (int(0.30*W), int(1.00*rowH), int(0.95*W), int(1.90*rowH)),
            "GER40":  (int(0.30*W), int(2.00*rowH), int(0.95*W), int(2.90*rowH)),
            "XAUUSD": (int(0.30*W), int(3.00*rowH), int(0.95*W), int(3.90*rowH)),
        }
        for sym, r in fallback.items():
            boxes.setdefault(sym, r)
    return boxes

# ---------- 価格読み取り（ROI内） ----------
def _read_price_roi(sym: str, roi: np.ndarray) -> Optional[float]:
    # 1) Tesseract（速い）
    val = _clean_num(_tesseract_number(roi) or "")
    if _valid(sym, val):
        return val

    # 2) 2値化バリエーション×Tesseract
    variants = []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    for th in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
        _, b = cv2.threshold(gray, 0, 255, th + cv2.THRESH_OTSU)
        variants.append(b)
    for v in variants:
        val = _clean_num(_tesseract_number(v) or "")
        if _valid(sym, val):
            return val

    # 3) EasyOCR（やや遅いが強い）
    reader = _easyocr_reader()
    res = reader.readtext(roi, detail=1, paragraph=False, decoder='beamsearch')
    # 数字っぽい候補をスコア順に精査
    for (bbox, text, conf) in res:
        val = _clean_num(str(text))
        if _valid(sym, val):
            return val
    return None

# ---------- エントリポイント ----------
def extract_watchlist_mobile_v13(img_path_or_pil) -> Dict[str, Optional[float]]:
    img = _load_image(img_path_or_pil)
    boxes = _find_label_boxes(img)
    cv_img = _to_cv(img)

    out: Dict[str, Optional[float]] = {"JP225": None, "NAS100": None, "GER40": None, "XAUUSD": None}
    for sym, (x1,y1,x2,y2) in boxes.items():
        roi = cv_img[max(0,y1):y2, max(0,x1):x2]
        out[sym] = _read_price_roi(sym, roi)
    out["engine"] = "mobile-v13-easyocr+tesseract-labeled"
    return out
