import os, time, json, re
from typing import Dict, Any, Optional
import requests

CACHE_PATH = "news_cache.json"
CACHE_TTL  = 300  # 5分

# シンボル→ニュース検索対象の説明
SYMBOL_META = {
    "JP225":  {"query": "Nikkei 225 futures short-term outlook (next 2-6h)", "asset": "JP225"},
    "NAS100": {"query": "NASDAQ 100 futures short-term outlook (next 2-6h)", "asset": "NAS100"},
    "GER40":  {"query": "DAX (GER40) futures short-term outlook (next 2-6h)", "asset": "GER40"},
    "XAUUSD": {"query": "Gold XAUUSD short-term outlook (next 2-6h)", "asset": "XAUUSD"},
}

def _load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_PATH):
        try:
            return json.load(open(CACHE_PATH, "r"))
        except Exception:
            return {}
    return {}

def _save_cache(obj: Dict[str, Any]) -> None:
    try:
        json.dump(obj, open(CACHE_PATH, "w"), ensure_ascii=False, indent=2)
    except Exception:
        pass

def _json_from_text(txt: str) -> Dict[str, Any]:
    # コードブロックごと/行頭余白を許容してJSON抽出
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def _pplx(symbol: str) -> Dict[str, Any]:
    key = os.getenv("PPLX_API_KEY")
    if not key:
        return {"direction": None, "confidence": 0.0, "summary": "(No PPLX API key)"}
    url = "https://api.perplexity.ai/chat/completions"
    prompt = f"""
You are a trading news assistant. Return ONLY compact JSON like:
{{"direction":"BUY|SELL|NEUTRAL","confidence":0.0-1.0,"summary":"..."}}
Task: Based on the latest news, macro, and futures tape for {symbol}, 2-6h horizon.
Consider market breadth, yields, FX, and scheduled events. Be concise.
"""
    body = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 300,
    }
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {key}",
                                        "Content-Type":"application/json"},
                          json=body, timeout=20)
        r.raise_for_status()
        data = r.json()
        txt  = data["choices"][0]["message"]["content"]
        js   = _json_from_text(txt)
        return {
            "direction": (js.get("direction") or js.get("dir")),
            "confidence": float(js.get("confidence", 0.0)),
            "summary": js.get("summary", txt)[:800],
            "raw": txt[:1500],
        }
    except Exception as e:
        return {"direction": None, "confidence": 0.0, "summary": f"(PPLX error: {e})"}

def _gpt(symbol: str) -> Dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {"direction": None, "confidence": 0.0, "summary": "(No OpenAI API key)"}
    url = "https://api.openai.com/v1/chat/completions"
    prompt = f"""
Return ONLY JSON: {{"direction":"BUY|SELL|NEUTRAL","confidence":0.0-1.0,"summary":"..."}}
Context: short-term (2-6h) outlook for {symbol}. Use global news, macro prints, yields, risk sentiment.
Be decisive but not reckless; keep it strictly for intraday positioning.
"""
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 300,
    }
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {key}",
                                        "Content-Type":"application/json"},
                          json=body, timeout=20)
        r.raise_for_status()
        data = r.json()
        txt  = data["choices"][0]["message"]["content"]
        js   = _json_from_text(txt)
        return {
            "direction": (js.get("direction") or js.get("dir")),
            "confidence": float(js.get("confidence", 0.0)),
            "summary": js.get("summary", txt)[:800],
            "raw": txt[:1500],
        }
    except Exception as e:
        return {"direction": None, "confidence": 0.0, "summary": f"(GPT error: {e})"}

def _dir_to_num(d: Optional[str]) -> float:
    if not d: return 0.0
    d = d.upper()
    return 1.0 if d == "BUY" else (-1.0 if d == "SELL" else 0.0)

def _consensus(pplx: Dict[str, Any], gpt: Dict[str, Any]) -> Dict[str, Any]:
    d1, d2 = pplx.get("direction"), gpt.get("direction")
    n1, n2 = _dir_to_num(d1), _dir_to_num(d2)

    if n1 == n2:
        agree = 1.0 if n1 != 0 else 0.66
    elif n1 == 0 or n2 == 0:
        agree = 0.5
    else:
        agree = 0.0

    conf = (float(pplx.get("confidence", 0.0)) + float(gpt.get("confidence", 0.0))) / 2.0

    final = None
    if d1 and d2 and d1.upper() == d2.upper() and d1.upper() != "NEUTRAL":
        final = d1.upper()
    else:
        cand = sorted(
            [("PPLX", d1, pplx.get("confidence",0.0)), ("GPT", d2, gpt.get("confidence",0.0))],
            key=lambda x: float(x[2]),
            reverse=True
        )
        for _, dd, _c in cand:
            if dd and dd.upper() in ("BUY","SELL"):
                final = dd.upper(); break
        if not final: final = "NEUTRAL"

    return {"direction": final, "agreement": agree, "confidence": conf}

def news_consensus(symbol: str) -> Dict[str, Any]:
    symbol = symbol.upper()
    meta = SYMBOL_META.get(symbol)
    if not meta:
        raise ValueError(f"Unsupported symbol: {symbol}")

    cache = _load_cache()
    now = time.time()
    key = f"{symbol}"

    if key in cache and now - cache[key].get("ts", 0) < CACHE_TTL:
        return cache[key]["data"]

    pplx = _pplx(symbol)
    gpt  = _gpt(symbol)
    con  = _consensus(pplx, gpt)
    data = {
        "symbol": symbol,
        "direction": con["direction"],
        "agreement": float(con["agreement"]),
        "confidence": float(con["confidence"]),
        "pplx": pplx,
        "gpt":  gpt,
        "ts":   now,
    }
    cache[key] = {"ts": now, "data": data}
    _save_cache(cache)
    return data
