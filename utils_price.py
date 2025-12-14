# utils_price.py
from typing import Dict, Any, List
import math

# Tick設定
TICK = {
    "JP225": 5.0,
    "NAS100": 0.25,
    "GER40": 0.5,
    "XAUUSD": 0.05
}

DECIMALS = {
    "JP225": 0,
    "NAS100": 2,
    "GER40": 1,
    "XAUUSD": 2
}

def _fmt(sym: str, v: float) -> float:
    d = DECIMALS[sym]
    return float(f"{v:.{d}f}")

def _round_tick(sym: str, v: float) -> float:
    t = TICK[sym]
    return _fmt(sym, round(v / t) * t)

def generate_ifd_for_side(sym: str, price: float, side: str,
                          strong_tp: bool = False) -> List[Any]:
    """
    strong_tp=True → STRONG_GO（強気利確）を反映した TP2 を生成
    """

    t = TICK[sym]

    # entry の距離（三tick）
    if side == "BUY":
        entry = _round_tick(sym, price - 3*t)
        sl = _fmt(sym, entry - 10*t)
        tp1 = _fmt(sym, entry + 6*t)

        # STRONG_GO の場合 TP2 を広げる
        if strong_tp:
            # 銘柄ごとの拡大幅
            if sym == "JP225":
                tp2 = _fmt(sym, entry + 1500)
            elif sym in ("NAS100", "GER40"):
                tp2 = _fmt(sym, entry + 15*t)
            elif sym == "XAUUSD":
                tp2 = _fmt(sym, entry + 8*t)
        else:
            tp2 = _fmt(sym, entry + 10*t)

    else:  # SELL
        entry = _round_tick(sym, price + 3*t)
        sl = _fmt(sym, entry + 10*t)
        tp1 = _fmt(sym, entry - 6*t)

        if strong_tp:
            if sym == "JP225":
                tp2 = _fmt(sym, entry - 1500)
            elif sym in ("NAS100", "GER40"):
                tp2 = _fmt(sym, entry - 15*t)
            elif sym == "XAUUSD":
                tp2 = _fmt(sym, entry - 8*t)
        else:
            tp2 = _fmt(sym, entry - 10*t)

    return [
        "DAY6H", sym, side, entry, sl, tp1, tp2,
        "指値", "ENTRY_OK", False, "", 1,
        "SMA25<SMA75 or MACD<Signal"
    ]

def build_ifd_rows(prices: Dict[str, float], strong_tp: bool = False):
    """BUY/SELL の両方の IFD 行を返す"""
    rows = []
    for sym, price in prices.items():
        if sym not in TICK:        # ← これを追加（engine 等を弾く）
            continue
        if price is None:
            continue
        rows.append(generate_ifd_for_side(sym, price, "BUY", strong_tp))
        rows.append(generate_ifd_for_side(sym, price, "SELL", strong_tp))
    return rows

def build_ifdoco_lines(rows: List[List[Any]]) -> str:
    out = []
    for r in rows:
        _, sym, side, entry, sl, tp1, tp2, *_rest = r
        stars = r[10]
        out.append(f"IFDOCO {sym} {side} 指値 entry={entry} SL={sl} TP={tp1}/{tp2} {stars}")
    return "\n".join(out)
