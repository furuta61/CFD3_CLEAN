# ================================
# HyperOptic v13 (robust, mobile-only)
# iPhone/スマホ縦4銘柄: JP225 / NAS100 / GER40 / XAUUSD
# ・PaddleOCR 新旧戻り値どちらでもOK
# ・数字のみ抽出 + 銘柄レンジでノイズ除去
# ・縦スライド/高さスイープ + 複数前処理で強力に拾いにいく
# ================================
import os, re, cv2, numpy as np
from typing import Dict, List, Tuple, Optional
from paddleocr import PaddleOCR

# --- トグル ---
DEBUG_SAVE = False           # Trueにすると各ROI/各前処理の画像を roidebug_v13/ に保存
DEBUG_DIR  = "roidebug_v13"

# --- OCR初期化（文書補正OFF/角度分類OFF） ---
ocr = PaddleOCR(lang="en", use_angle_cls=False)

# --- スマホ基準ROI（1320×2868基準）。まず確実に数字帯を含むよう高さを広めに ---
BASE_ROI = {
    "JP225":  (430,  520, 890, 320),   # (x, y, w, h)
    "NAS100": (430,  870, 890, 320),
    "GER40":  (430, 1210, 890, 320),
    "XAUUSD": (430, 1550, 890, 320),
}

# --- 妥当レンジ（数値のみ残すためのフィルタ） ---
RANGE = {
    "JP225":  (10_000, 80_000),
    "NAS100": ( 8_000, 50_000),
    "GER40":  ( 8_000, 50_000),
    "XAUUSD": ( 1_000,  5_000),
}

_num_pat = re.compile(r"\d+(?:\.\d+)?")

def _rescale_rect(xywh: Tuple[int,int,int,int], w: int, h: int) -> Tuple[int,int,int,int]:
    """1320x2868基準のROIを実サイズへスケール & 画面内にクリップ"""
    x, y, rw, rh = xywh
    X1 = int(x * w / 1320.0)
    Y1 = int(y * h / 2868.0)
    X2 = int((x + rw) * w / 1320.0)
    Y2 = int((y + rh) * h / 2868.0)
    X1 = max(0, min(X1, w-1)); X2 = max(0, min(X2, w))
    Y1 = max(0, min(Y1, h-1)); Y2 = max(0, min(Y2, h))
    return X1, Y1, max(1, X2 - X1), max(1, Y2 - Y1)

def _prep_variants(gray: np.ndarray) -> List[np.ndarray]:
    """複数前処理を用意（どれか当たればOK）"""
    outs = []
    # 1) コントラスト強調
    g1 = cv2.convertScaleAbs(gray, alpha=1.6, beta=10)
    outs.append(g1)
    # 2) 反転 + 固定二値（閾値強め/弱め）
    inv = cv2.bitwise_not(g1)
    th180 = cv2.threshold(inv, 180, 255, cv2.THRESH_BINARY)[1]
    outs.append(th180)
    th160 = cv2.threshold(inv, 160, 255, cv2.THRESH_BINARY)[1]
    outs.append(th160)
    # 3) CLAHE
    try:
        cla = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
        outs.append(cla)
    except Exception:
        outs.append(cv2.equalizeHist(gray))
    # 4) 適応二値（局所コントラストに強い）
    adap1 = cv2.adaptiveThreshold(
        cv2.bitwise_not(gray), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, 8
    )
    outs.append(adap1)
    return outs

def _texts_scores_from_paddle(res) -> List[Tuple[str, float]]:
    """PaddleOCRの新旧出力に対応して (text, score) の配列に正規化"""
    out: List[Tuple[str,float]] = []
    if not res:
        return out
    # 新API: [ { 'rec_texts': [...], 'rec_scores': [...] , ... } ]
    if isinstance(res[0], dict):
        texts  = res[0].get("rec_texts", []) or []
        scores = res[0].get("rec_scores", []) or []
        for t, s in zip(texts, scores):
            if isinstance(t, str):
                out.append((t, float(s)))
        return out
    # 旧API: [ [ [box, (text, score)], ... ], ... ]
    for line in res:
        if not isinstance(line, list):
            continue
        for item in line:
            txt, sc = None, 0.0
            if isinstance(item, list) and len(item) >= 2:
                meta = item[1]
                if isinstance(meta, (list, tuple)) and len(meta) >= 2 and isinstance(meta[0], str):
                    txt, sc = meta[0], float(meta[1])
                elif isinstance(meta, str):
                    txt = meta; sc = 0.0
            if isinstance(txt, str):
                out.append((txt, sc))
    return out

def _pick_number(cands: List[Tuple[float,float]], name: str) -> Optional[float]:
    """(value, score)から銘柄レンジ内の最良を選ぶ。無ければ桁＋scoreで最大を選択。"""
    lo, hi = RANGE.get(name, (0, float("inf")))
    in_range = [(v,s) for (v,s) in cands if lo <= v <= hi]
    if in_range:
        in_range.sort(key=lambda x: (x[1], len(str(int(x[0])))), reverse=True)
        return in_range[0][0]
    if cands:
        # 桁数優先でスコアを補助指標にする
        cands.sort(key=lambda x: (len(str(int(x[0]))), x[1]))
        return cands[-1][0]
    return None

def _scan_roi(img: np.ndarray, rect: Tuple[int,int,int,int], name: str, tag: str) -> Optional[float]:
    """ROIについて縦スライド & 高さスイープしながら複数前処理を試す"""
    x, y, w, h = rect
    H, W = img.shape[:2]

    # 縦にスライド（±80px相当）、高さ拡張（1.0/1.25/1.5）
    offsets  = [ -80, -40, 0, 40, 80 ]
    heights  = [ 1.00, 1.25, 1.50 ]

    best: List[Tuple[float,float]] = []
    dbg_idx = 0

    for dy in offsets:
        for mh in heights:
            Y  = max(0, min(H-1, y + dy))
            Hd = int(min(H - Y, h * mh))
            roi = img[Y:Y+Hd, x:x+w]
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            preps = _prep_variants(gray)
            for i, prep in enumerate(preps):
                # デバッグ保存
                if DEBUG_SAVE:
                    os.makedirs(DEBUG_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(
                        DEBUG_DIR, f"{tag}_{name}_{dbg_idx:03d}.png"
                    ), prep)
                    dbg_idx += 1

                # グレースケールならBGRに変換（PaddleOCRは3ch期待）
                if len(prep.shape) == 2:
                    prep = cv2.cvtColor(prep, cv2.COLOR_GRAY2BGR)

                res = ocr.ocr(prep)        # ここはAPI差異を意識しない
                ts  = _texts_scores_from_paddle(res)

                for t, s in ts:
                    for m in _num_pat.findall(t):
                        try:
                            v = float(m)
                            best.append((v, float(s)))
                        except:
                            pass

    return _pick_number(best, name)

def extract_watchlist_mobile_v13(path: str) -> Dict[str, Optional[float]]:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("画像を読み込めませんでした。パスを確認してください。")
    h, w = img.shape[:2]

    results: Dict[str, Optional[float]] = {}
    for name, xywh in BASE_ROI.items():
        rect = _rescale_rect(xywh, w, h)
        val  = _scan_roi(img, rect, name, tag=os.path.basename(path).split('.')[0])
        results[name] = val

    results["engine"] = "mobile-v13"
    return results

# ROI確認用（希望時のみ使用）
def save_rois_v13(path: str, outdir: str = "roidebug_v13_base") -> None:
    os.makedirs(outdir, exist_ok=True)
    img = cv2.imread(path)
    h, w = img.shape[:2]
    for name, xywh in BASE_ROI.items():
        x, y, rw, rh = _rescale_rect(xywh, w, h)
        cv2.imwrite(os.path.join(outdir, f"roi_{name}.png"), img[y:y+rh, x:x+rw])
