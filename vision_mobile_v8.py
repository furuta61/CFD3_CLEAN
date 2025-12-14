# vision_mobile_v8.py
#
# iPhone GMOアプリ「ウォッチリスト」スクショ専用
# 4銘柄（JP225 / NAS100 / GER40 / XAUUSD）の現在値だけを抽出する v8 エンジン
#
# 必要ライブラリ: Pillow, pytesseract
# pip install pillow pytesseract

import io
import re
from typing import Dict, Optional

from PIL import Image, ImageOps, ImageFilter
import pytesseract


TICKERS = ["JP225", "NAS100", "GER40", "XAUUSD"]


def _preprocess_price_region(im: Image.Image) -> Image.Image:
    """価格領域を OCR しやすくする前処理（cv2 なし版）"""
    # グレースケール
    gray = im.convert("L")
    # コントラスト強調（白文字をより白く）
    gray = ImageOps.autocontrast(gray)
    # 少しシャープに
    gray = gray.filter(ImageFilter.SHARPEN)
    # 2倍に拡大（小さい数字対策）
    w, h = gray.size
    gray = gray.resize((w * 2, h * 2), Image.BICUBIC)
    return gray


def _ocr_price(im: Image.Image) -> Optional[float]:
    """価格領域から「一番それっぽい価格」を1つ返す"""
    proc = _preprocess_price_region(im)

    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789."
    text = pytesseract.image_to_string(proc, config=cfg)
    text = text.strip()

    if not text:
        return None

    # 3桁以上の整数部を持つ数値だけを候補にする（ノイズ除外）
    candidates = re.findall(r"\d{3,6}(?:\.\d+)?", text)
    if not candidates:
        return None

    # 最後に出てきた値を採用（大きな価格が最後に出ることが多い）
    try:
        vals = [float(c.replace(",", "")) for c in candidates]
        # 不自然な値は除外（JP225/NAS/GER/Gold を想定したゆるいレンジ）
        vals = [v for v in vals if 1000.0 <= v <= 100000.0]
        if not vals:
            return None
        return vals[-1]
    except Exception:
        return None


def extract_watchlist_mobile_v8(image_bytes: bytes) -> Dict[str, Optional[float]]:
    """
    iPhone GMOウォッチリスト（4銘柄縦並び）のスクショから、
    4銘柄の現在値だけを抽出する。

    戻り値:
        {"JP225": float|None, "NAS100": float|None,
         "GER40": float|None, "XAUUSD": float|None,
         "engine": "fixedroi-mobile-v8"}
    """
    im = Image.open(io.BytesIO(image_bytes))
    w, h = im.size

    # 上下の余白をざっくり除外（ステータスバー + タブ / 下部ナビ）
    top_margin = int(h * 0.13)   # 上側メニュー類
    bottom_margin = int(h * 0.10)  # 下メニュー

    usable_h = h - top_margin - bottom_margin

    # 4行に均等分割して、各行の中央付近だけを見る
    row_height = usable_h / 4.0

    # 価格は画面の右寄り中央にあるので、横 35%〜90% を価格候補にする
    x0 = int(w * 0.35)
    x1 = int(w * 0.90)

    out: Dict[str, Optional[float]] = {t: None for t in TICKERS}

    for idx, ticker in enumerate(TICKERS):
        row_y_center = top_margin + int(row_height * (idx + 0.5))
        y0 = int(row_y_center - row_height * 0.25)
        y1 = int(row_y_center + row_height * 0.25)

        # 安全のため画像範囲内にクリップ
        y0 = max(0, y0)
        y1 = min(h, y1)

        crop = im.crop((x0, y0, x1, y1))

        price = _ocr_price(crop)
        out[ticker] = price

    out["engine"] = "fixedroi-mobile-v8"
    return out
