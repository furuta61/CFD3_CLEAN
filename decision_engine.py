# decision_engine.py
from typing import Dict, Any, Tuple, List
import time
from utils_price import build_ifd_rows
from news_signal import news_consensus

SYMBOLS = ("JP225", "NAS100", "GER40", "XAUUSD")

def _clean_prices(prices: dict) -> dict:
    """4銘柄のみ＆数値のみ通す（engineなどを除去）"""
    ok = {}
    for s in SYMBOLS:
        v = prices.get(s)
        if isinstance(v, (int, float)):
            ok[s] = float(v)
    return ok

def _score_and_stars(agreement: float, confidence: float) -> Tuple[float, str]:
    score = agreement * 0.6 + confidence * 0.4
    if score >= 0.90: stars = "★★★★★"
    elif score >= 0.80: stars = "★★★★☆"
    elif score >= 0.70: stars = "★★★☆☆"
    elif score >= 0.60: stars = "★★☆☆☆"
    else: stars = "★☆☆☆☆"
    return score, stars

def _stars_to_signal(stars: str) -> str:
    # 表の「判定」欄に出す信号
    table = {
        "★★★★★": "STRONG_GO",
        "★★★★☆": "GO",
        "★★★☆☆": "WEAK_GO",
        "★★☆☆☆": "HOLD",
        "★☆☆☆☆":  "HOLD",
    }
    return table.get(stars, "HOLD")

def _dir_en(d: str | None) -> str:
    if d == "BUY":  return "buy"
    if d == "SELL": return "sell"
    return "観察"

def _load_news(symbol: str) -> Dict[str, Any]:
    ns = news_consensus(symbol)  # {direction, agreement, confidence, pplx, gpt}
    score, stars = _score_and_stars(ns.get("agreement", 0.0),
                                    ns.get("confidence", 0.0))
    return {"dir": ns.get("direction"),
            "score": score, "stars": stars, "raw": ns}

CUT_TXT = "SMA25＜SMA75 または MACD＜Signal"

def manual_table_rows(prices: Dict[str, float],
                      lots: int = 1,
                      trade_mode: str = "DAY6H") -> tuple[List[List[Any]], Dict[str, Any]]:
    """
    ニュース合意を使って各銘柄＝片側1本に絞り、**必ず表を4行**返す。
    方向が取れなければ「観察」行（価格は'-'）を出す。
    戻り値 rows は、以下13列で固定：
    [trade_mode, 銘柄, 方向, entry_price, SL, TP1, TP2, order_type, 判定, ニュースロック, 推奨度, ロット, CUT条件]
    """
    prices = _clean_prices(prices)
    # BUY/SELL両側の素テーブル（価格生成用）
    base = build_ifd_rows(prices, strong_tp=False)
    base_map = {(r[1], r[2]): r for r in base}  # (sym, 'BUY'/'SELL') -> row
    rows: List[List[Any]] = []
    summaries: Dict[str, Any] = {}

    for sym in SYMBOLS:
        if sym not in prices:
            # OCRに無い銘柄はスキップ（行は出さない）
            continue
        info = _load_news(sym)
        summaries[sym] = info
        want = info["dir"]  # "BUY"/"SELL"/None

        if want in ("BUY", "SELL") and (sym, want) in base_map:
            r = base_map[(sym, want)]
            entry, sl, tp1 = r[3], r[4], r[5]  # TP2は表では '-' にする仕様
            stars = info["stars"]
            signal = _stars_to_signal(stars)
            rows.append([
                trade_mode, sym, _dir_en(want),
                entry, sl, tp1, "-",      # TP2はハイフン固定
                "指値", signal, "false",
                stars, lots, CUT_TXT
            ])
        else:
            # 方向が取れない → 観察行（数値は '-')
            rows.append([
                trade_mode, sym, "観察",
                "-", "-", "-", "-",
                "指値", "HOLD", "false",
                "★☆☆☆☆", 0, CUT_TXT
            ])

    return rows, {"mode": "manual", "summaries": summaries}

def auto_table_rows(prices: Dict[str, float],
                    gate: float = 0.90,
                    lots: int = 1,
                    trade_mode: str = "SWING1") -> tuple[List[List[Any]], Dict[str, Any]]:
    """
    TV（tv_latest.json）×ニュース一致スコアで、**該当銘柄1行のみ**返す。
    閾値未達なら1行の観察行を返す（空では返さない）。
    """
    import os, json
    prices = _clean_prices(prices)
    if not os.path.exists("tv_latest.json"):
        # TVが無くても1行は返す（観察）
        return ([[trade_mode, "—", "観察", "-", "-", "-", "-",
                  "指値", "HOLD", "false", "★☆☆☆☆", 0, CUT_TXT]],
                {"mode": "auto", "error": "No TV alert yet."})

    tv = json.load(open("tv_latest.json"))
    sym = tv.get("symbol")
    if not sym or sym not in prices:
        return ([[trade_mode, tv.get("symbol","—"), "観察", "-", "-", "-", "-",
                  "指値", "HOLD", "false", "★☆☆☆☆", 0, CUT_TXT]],
                {"mode": "auto", "error": "TV symbol invalid or not in OCR set.", "tv": tv})

    info = _load_news(sym)
    proceed = (info["dir"] in ("BUY", "SELL")) and (info["score"] >= gate)

    if proceed:
        base = build_ifd_rows({sym: prices[sym]}, strong_tp=False)
        pick = [r for r in base if r[1] == sym and r[2] == info["dir"]]
        if pick:
            r = pick[0]
            entry, sl, tp1 = r[3], r[4], r[5]
            stars = info["stars"]
            signal = _stars_to_signal(stars)
            rows = [[
                trade_mode, sym, _dir_en(info["dir"]),
                entry, sl, tp1, "-",
                "指値", signal, "false", stars, lots, CUT_TXT
            ]]
        else:
            rows = [[trade_mode, sym, "観察", "-", "-", "-", "-",
                     "指値", "HOLD", "false", "★☆☆☆☆", 0, CUT_TXT]]
    else:
        rows = [[trade_mode, sym, "観察", "-", "-", "-", "-",
                 "指値", "HOLD", "false", "★☆☆☆☆", 0, CUT_TXT]]

    meta = {"mode": "auto", "tv": tv, "score": info["score"],
            "stars_txt": info["stars"],
            "verdict": "PROCEED" if proceed else "SKIP",
            "gate": gate, "news": info["raw"]}
    return rows, meta


def news_consensus_all() -> Dict[str, Dict[str, Any]]:
    """4銘柄のニュース短観をまとめて返す。"""
    # simple in-memory TTL cache to avoid repeated external calls
    try:
        _cache = news_consensus_all._cache
    except AttributeError:
        _cache = news_consensus_all._cache = {"ts": 0, "data": {}}

    TTL = 120.0  # seconds
    now = time.time()
    if now - _cache.get("ts", 0) <= TTL and _cache.get("data"):
        return _cache["data"]

    symbols = ["JP225", "NAS100", "GER40", "XAUUSD"]
    out: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        info = news_consensus(sym)
        score, stars = _score_and_stars(info.get("agreement", 0.0), info.get("confidence", 0.0))
        out[sym] = {**info, "score": score, "stars": stars}

    _cache["ts"] = now
    _cache["data"] = out
    return out

