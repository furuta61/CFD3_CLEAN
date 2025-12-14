# ==========================================================
#  vision_mobile_v11.py  (スマホ4銘柄専用 / Tesseract版)
#  - GMOスマホウォッチリスト縦4行から
#    JP225, NAS100, GER40, XAUUSD の現在値だけを一括抽出
# ==========================================================

import cv2
import numpy as np
import pytesseract


def _extract_big_prices(img):
    """
    画面全体から「大きな数字フォント」を4つ抽出する。
    1. pytesseract.image_to_data で全テキスト取得
    2. 数字っぽいトークンだけ残す
    3. その中で「高さが最大の文字」を基準に、上位の大きいものだけ選ぶ
    """
    h, w = img.shape[:2]

    data = pytesseract.image_to_data(
        img,
        lang="jpn+eng",
        config="--psm 6 --oem 3",
        output_type=pytesseract.Output.DICT,
    )

    n = len(data["text"])
    candidates = []

    for i in range(n):
        raw = data["text"][i]
        if not raw or raw.strip() == "":
            continue

        # 数字 + 小数点だけ残す
        cleaned = "".join(ch for ch in raw if (ch.isdigit() or ch == "."))
        if cleaned.count(".") > 1:
            continue
        if len(cleaned.replace(".", "")) < 4:  # 3桁以下はノイズ（0.7, 3.0など）
            continue

        try:
            val = float(cleaned)
        except Exception:
            continue

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = 0.0

        candidates.append(
            {
                "val": val,
                "clean": cleaned,
                "conf": conf,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            }
        )

    if not candidates:
        return []

    # 一番大きいフォントサイズ（height）を基準にフィルタ
    max_h = max(c["height"] for c in candidates)
    thresh = max_h * 0.75  # 上位 25% 程度に絞る

    big = [c for c in candidates if c["height"] >= thresh]

    # それでも4未満なら、高さ順で上位4つに補完
    if len(big) < 4:
        big = sorted(candidates, key=lambda c: c["height"], reverse=True)[:4]

    # 画面上の位置（top）で並べて、上から JP225, NAS100, GER40, XAUUSD に割当
    big_sorted = sorted(big, key=lambda c: c["top"])
    return big_sorted[:4]


def extract_watchlist_mobile_v11(img_path: str):
    """
    スマホのウォッチリストスクショ（縦4銘柄）から
    JP225, NAS100, GER40, XAUUSD の price を一括抽出。

    戻り値:
      {
        "JP225": float | None,
        "NAS100": float | None,
        "GER40": float | None,
        "XAUUSD": float | None,
        "engine": "mobile-v11-tess"
      }
    """
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"画像が読めませんでした: {img_path}")

    big_prices = _extract_big_prices(img)

    # いったん None で初期化
    result = {
        "JP225": None,
        "NAS100": None,
        "GER40": None,
        "XAUUSD": None,
        "engine": "mobile-v11-tess",
    }

    tickers = ["JP225", "NAS100", "GER40", "XAUUSD"]

    for ticker, cand in zip(tickers, big_prices):
        result[ticker] = cand["val"]

    return result


