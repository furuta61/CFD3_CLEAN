
# -*- coding: utf-8 -*-
# v13-easyocr-ycluster: GMOウォッチリスト用の堅牢OCR
# 手順: 画像全体をOCR → 数字のみ抽出 → y座標で4クラスタに自動分割 → 各行で一番大きい数字の箱を採用
# 依存: easyocr, opencv-python, numpy

from typing import Dict, List, Tuple, Optional
import numpy as np, cv2, re
import easyocr

ENGINE_NAME = "mobile-v13-easyocr-ycluster+gold-fallback2"

SYMBOLS = ["JP225", "NAS100", "GER40", "XAUUSD"]  # 画面の上→下
# 表示の小数桁（GMOの見え方に合わせて丸め）
DECIMALS = {"JP225": 1, "NAS100": 1, "GER40": 1, "XAUUSD": 2}
# 値域（異常値除外）
RANGE = {
    "JP225":  (15000, 70000),
    "NAS100": ( 5000, 50000),
    "GER40":  (10000, 30000),
    "XAUUSD": ( 3000, 10000),  # 金スポットは 4xxx〜5xxx 台
}

MIN_H_FRAC = 0.015          # 文字高さの最小割合（従来 0.02 → もっと小さい文字も拾う）
BOTTOM_SALVAGE_Y = 0.72     # 画面下 72% 以降を「金スポット」候補ゾーンとして救済

_reader: Optional[easyocr.Reader] = None


# --- numeric normalizer (共通) ---
def _to_float_safe(txt: str):
    if not txt:
        return None
    s = txt.strip()
    s = (s.replace(',', '')
           .replace('·', '.').replace('•', '.')
           .replace('：', ':').replace(':', '.').replace('。', '.')
           .replace('O', '0').replace('o', '0')
           .replace('I', '1').replace('l', '1'))
    s = ''.join(re.findall(r'[0-9\.]+', s))
    if not s or s == '.':
        return None
    if s.count('.') > 1:
        left, _, right = s.rpartition('.')
        s = left.replace('.', '') + '.' + right
    try:
        v = float(s)
        if v > 10000 and v < 1000000:
            v = v / 100.0
        return v
    except Exception:
        return None


def _get_reader() -> easyocr.Reader:
    """Lazy-load EasyOCR reader (ja+en, CPU)."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['ja', 'en'], gpu=False, verbose=False)
    return _reader


def _to_bgr(path_or_bgr):
    if isinstance(path_or_bgr, str):
        data = np.fromfile(path_or_bgr, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        img = path_or_bgr
    return img


def _preprocess(img: np.ndarray) -> np.ndarray:
    # 周縁トリムを少し緩める（下端を削りすぎない）
    h, w = img.shape[:2]
    img = img[int(0.05 * h):int(0.995 * h), int(0.02 * w):int(0.995 * w)].copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return img


_num_pat = re.compile(r"^[\-]?[0-9]{2,3}(?:[, ]?[0-9]{3})*(?:\.[0-9]{1,3})?$")


def _clean_digits(s: str) -> str:
    # よくある誤認を補正
    s = s.strip()
    s = s.replace("O", "0").replace("o", "0").replace("S", "5").replace("s", "5").replace("B", "8")
    s = s.replace("’", "").replace("`", "").replace(" ", "")
    return s


def _parse_float(text: str) -> Optional[float]:
    t = _clean_digits(text)
    if not _num_pat.match(t):
        return None
    t = t.replace(",", "")
    try:
        return float(t)
    except Exception:
        return None


def _fallback_gold_roi(img):
    h, w = img.shape[:2]
    x0, x1 = int(0.28 * w), int(0.78 * w)
    y0, y1 = int(0.73 * h), int(0.87 * h)
    roi = img[y0:y1, x0:x1].copy()

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 20, 20)
    g = cv2.equalizeHist(g)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    return roi, bw


def _get_easyocr_reader():
    return _get_reader()


def _locate_gold_band_by_info(img):
    """
    EasyOCR で 'info' を全部拾い、最も下の bbox を返す。
    戻り値: (cx, cy, w, h)  or None
    """
    reader = _get_easyocr_reader()
    results = reader.readtext(img, detail=1, paragraph=False, min_size=8)
    infos = []
    for (bbox, text, conf) in results:
        t = (text or '').strip().lower()
        if 'info' in t and conf >= 0.3:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            cx = sum(xs) / 4.0
            cy = sum(ys) / 4.0
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            infos.append((cx, cy, w, h))
    if not infos:
        return None
    return sorted(infos, key=lambda t: t[1])[-1]


def _read_gold_from_band(img):
    """
    1) 一番下の 'info' を探す → その y を中心に帯を切る
    2) 帯の中で中央〜右側を ROI にし、pytesseract で数値のみ読む
    """
    import cv2
    import numpy as np
    import pytesseract

    h, w = img.shape[:2]
    anchor = _locate_gold_band_by_info(img)
    if anchor is None:
        return None
    _, cy, _, ih = anchor

    band_h = max(int(0.09 * h), int(2.5 * ih))
    y0 = max(0, int(cy - band_h // 2))
    y1 = min(h, y0 + band_h)

    x0 = int(0.22 * w)
    x1 = int(0.82 * w)
    band = img[y0:y1, x0:x1]

    try:
        import os
        os.makedirs("debug", exist_ok=True)
        cv2.imwrite("debug/gold_band.png", band)
    except Exception:
        pass

    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 25, 25)
    g = cv2.equalizeHist(g)
    bw1 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    _, bw2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cfg = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
    cand = []
    for m in (bw1, bw2, cv2.bitwise_not(bw1), cv2.bitwise_not(bw2)):
        txt = pytesseract.image_to_string(m, config=cfg).strip()
        v = _to_float_safe(txt)
        if v is not None:
            cand.append(v)

    best = None
    for v in cand:
        if 3500.0 <= v <= 5000.0:
            if (best is None) or (abs(v - 4300) < abs(best - 4300)):
                best = v

    if best is not None:
        return round(best, 2)
    return None


def _split_into_4_rows(boxes: List[Tuple[Tuple[int, int, int, int], float, float]], h_img: int):
    """
    boxes: [(x,y,w,h, center_y, box_h), value]
    yのギャップ最大3つで4分割
    """
    if not boxes:
        return [[], [], [], []]
    # y中心でソート
    boxes = sorted(boxes, key=lambda b: b[0][1] + b[0][3] / 2)
    ys = np.array([b[0][1] + b[0][3] / 2 for b in boxes], dtype=float)
    # 3つの最大ギャップをしきい値に
    gaps = np.diff(ys)
    if len(gaps) < 3:
        # 均等分割フォールバック
        cut1, cut2, cut3 = h_img * 0.25, h_img * 0.50, h_img * 0.75
    else:
        idx = np.argsort(gaps)[-3:]
        cuts = np.sort(ys[idx + 1])
        cut1, cut2, cut3 = cuts.tolist()
    rows = [[], [], [], []]
    for b in boxes:
        cy = b[0][1] + b[0][3] / 2
        k = 0 if cy < cut1 else (1 if cy < cut2 else (2 if cy < cut3 else 3))
        rows[k].append(b)
    return rows


def _pick_biggest_per_row(row_boxes):
    if not row_boxes:
        return None
    # フォントが一番大きい（box高さ最大）候補を採用
    row_boxes = sorted(row_boxes, key=lambda b: b[0][3], reverse=True)
    return row_boxes[0]


def extract_watchlist_mobile_v12(path_or_bgr, backend: Optional[str] = None, debug: bool = False, **_ignored) -> Dict[str, Optional[float]]:
    """
    返り値: {"JP225":float|None, "NAS100":..., "GER40":..., "XAUUSD":..., "engine":...}
    backendは互換性維持のため受け取るが本実装では未使用。
    """
    img = _to_bgr(path_or_bgr)
    if img is None:
        return {"JP225": None, "NAS100": None, "GER40": None, "XAUUSD": None, "engine": ENGINE_NAME}

    img = _preprocess(img)
    h, w = img.shape[:2]

    reader = _get_reader()
    # EasyOCRはBGR/RGBどちらもOKだが numpy配列を直接渡す
    results = reader.readtext(img, detail=1, paragraph=False)  # [(bbox, text, conf), ...]

    cand = []
    # bbox → (x,y,w,h)
    for bbox, text, conf in results:
        # bboxは4点 [[x1,y1],[x2,y2],...]
        x1, y1 = bbox[0]
        x3, y3 = bbox[2]
        x = int(min(x1, x3))
        y = int(min(y1, y3))
        ww = int(abs(x3 - x1))
        hh = int(abs(y3 - y1))
        if hh < h * MIN_H_FRAC:
            continue
        val = _parse_float(text)
        if val is None:
            continue
        cand.append(((x, y, ww, hh), val, conf))

    # yで4分割
    rows = _split_into_4_rows(cand, h)
    out: Dict[str, Optional[float]] = {s: None for s in SYMBOLS}

    for idx, sym in enumerate(SYMBOLS):
        best = _pick_biggest_per_row(rows[idx])
        if best is None:
            continue
        (_, _, _, _), v, _ = best
        # 値域チェック
        lo, hi = RANGE[sym]
        if not (lo <= v <= hi):
            for b in sorted(rows[idx], key=lambda z: z[0][3], reverse=True)[1:]:
                (_, _, _, _), vv, _ = b
                if lo <= vv <= hi:
                    v = vv
                    break
            else:
                continue
        out[sym] = float(f"{v:.{DECIMALS[sym]}f}")

    # --- ここから救済：XAUUSD が None のとき、画面下部で一番大きい数字を拾う ---
    if out["XAUUSD"] is None:
        bottom_cand = []
        lo, hi = RANGE["XAUUSD"]
        for (x, y, ww, hh), v, cf in cand:
            cy = y + hh / 2
            if cy >= h * BOTTOM_SALVAGE_Y and lo <= v <= hi:
                bottom_cand.append(((x, y, ww, hh), v, cf))
        if bottom_cand:
            best = sorted(bottom_cand, key=lambda z: z[0][3], reverse=True)[0]
            v = best[1]
            out["XAUUSD"] = float(f"{v:.{DECIMALS['XAUUSD']}f}")

    # --- XAUUSD が None のとき、専用ROIでもう一度だけ読む ---
    if out.get("XAUUSD") is None:
        roi = bw = None
        try:
            roi, bw = _fallback_gold_roi(img)
            reader = _get_reader()
            res1 = reader.readtext(roi, detail=0, allowlist='0123456789:.')
            res2 = reader.readtext(bw, detail=0, allowlist='0123456789:.')
            cand_txts = res1 + res2

            best = None
            for t in cand_txts:
                v = _to_float_safe(t)
                if v is None:
                    continue
                if 3500.0 <= v <= 5000.0:
                    if (best is None) or (abs(v - 4300) < abs(best - 4300)):
                        best = v
            if best is not None:
                out["XAUUSD"] = round(best, 2)
        except Exception:
            try:
                if roi is not None:
                    cv2.imwrite("gold_fallback_roi.png", roi)
                if bw is not None:
                    cv2.imwrite("gold_fallback_bw.png", bw)
            except Exception:
                pass

    if out.get("XAUUSD") is None:
        try:
            out["XAUUSD"] = _read_gold_from_band(img)
        except Exception:
            pass

    out["engine"] = ENGINE_NAME

    if any(out[s] is None for s in ["JP225", "NAS100", "GER40", "XAUUSD"]):
        dbg = img.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        rows = _split_into_4_rows(cand, h)
        for ridx, row in enumerate(rows):
            for (x, y, ww, hh), v, cf in row:
                cv2.rectangle(dbg, (x, y), (x + ww, y + hh), colors[ridx], 2)
        cv2.imwrite("ocr_debug_boxes.png", dbg)

    if debug:
        dbg = img.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for ridx, row in enumerate(rows):
            for (x, y, ww, hh), v, cf in row:
                cv2.rectangle(dbg, (x, y), (x + ww, y + hh), colors[ridx], 2)
        cv2.imwrite("ocr_debug_boxes.png", dbg)
    return out
    出力: {JP225, NAS100, GER40, XAUUSD, engine}
