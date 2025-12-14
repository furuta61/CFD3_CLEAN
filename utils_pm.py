from __future__ import annotations
from typing import Dict, Any, Optional
import json, os, datetime as dt

_STATE_FILE = "risk_state.json"

def _today_key() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")

def _load_state() -> Dict[str, Any]:
    if os.path.exists(_STATE_FILE):
        try:
            st = json.load(open(_STATE_FILE, "r"))
        except Exception:
            st = {}
    else:
        st = {}
    key = _today_key()
    if st.get("date") != key:
        st = {"date": key, "realized_yen": 0.0, "trades": 0}
    return st

def _save_state(st: Dict[str, Any]) -> None:
    json.dump(st, open(_STATE_FILE, "w"), ensure_ascii=False, indent=2)

def register_fill(pl_yen: float) -> Dict[str, Any]:
    st = _load_state()
    st["realized_yen"] = float(st.get("realized_yen", 0.0)) + float(pl_yen)
    st["trades"] = int(st.get("trades", 0)) + 1
    _save_state(st)
    return st

def current_state() -> Dict[str, Any]:
    return _load_state()

def allow_new_entry(target_day_yen: int, max_dday_yen: int) -> tuple[bool, Dict[str, Any]]:
    st = _load_state()
    pl = float(st.get("realized_yen", 0.0))
    # ターゲット達成済み → 取引停止
    if pl >= target_day_yen:
        return False, {"reason": "TARGET_HIT", "state": st}
    # ドローダウン到達 → 停止
    if pl <= -abs(max_dday_yen):
        return False, {"reason": "MAX_DD", "state": st}
    return True, {"reason": "OK", "state": st}

def suggest_lots_manual(stars: str, want_lots_strong: int = 5, want_lots_go: int = 2) -> int:
    # ★★★★★ or ★★★★☆ → 強、★★★☆☆ → GO、未満は見送り推奨
    strong = {"★★★★★", "★★★★☆"}
    go = {"★★★☆☆"}
    if stars in strong:
        return want_lots_strong
    if stars in go:
        return want_lots_go
    return 0  # 見送り

def clamp_lots_by_risk(remaining_dday_yen: float, sl_total_yen: float, lots: int) -> int:
    # 残り許容損失 ＜ 想定トレード損切り → ロット縮小
    if lots <= 0: 
        return 0
    if remaining_dday_yen <= 0:
        return 0
    if sl_total_yen <= 0:
        return lots
    # 1回負けたら終日終了しないよう、1/1.2 の安全係数で縮小
    safe_lots = int((remaining_dday_yen / 1.2) // (sl_total_yen / max(lots, 1)))
    return max(0, min(lots, safe_lots if safe_lots > 0 else 0))
