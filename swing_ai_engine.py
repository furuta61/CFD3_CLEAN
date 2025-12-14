# ============================================
# CFD3 Dual Engine - Swing AI (4 symbols batch)
# ============================================

import openai
import requests
import numpy as np

SYMBOLS = ["JP225", "NAS100", "GER40", "XAUUSD"]

# --------------------------------------------
# AI ニュース取得（GPT 3行）
# --------------------------------------------
def fetch_gpt_news(symbol: str) -> list[str]:
    prompt = f"""
あなたは金融アナリストです。以下の銘柄について、
その日の市場影響ニュースを「3行」で箇条書きにしてください。

銘柄: {symbol}

条件:
- リスクオン/リスクオフに関連する内容のみ
- テクニカル情報は含めない
- 簡潔に
"""
    res = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    text = res["choices"][0]["message"]["content"]
    return [line.strip("-• ") for line in text.split("\n") if line.strip()][:3]

# --------------------------------------------
# Perplexity ニュース（3行）
# --------------------------------------------
def fetch_px_news(symbol: str) -> list[str]:
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {YOUR_PERPLEXITY_API_KEY}"}
    payload = {
        "model": "sonar-small-online",
        "messages": [
            {
                "role": "user",
                "content": f"""
最新の市況ニュースから、{symbol} に影響する内容だけを 3行でまとめてください。
必ず要因だけを端的に。
"""
            }
        ]
    }
    res = requests.post(url, headers=headers, json=payload)
    text = res.json()["choices"][0]["message"]["content"]
    return [line.strip("-• ") for line in text.split("\n") if line.strip()][:3]

# --------------------------------------------
# 一致率の計算（GPTとPerplexityの類似度）
# --------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calc_consistency(gpt_lines: list[str], px_lines: list[str]) -> int:
    docs = [" ".join(gpt_lines), " ".join(px_lines)]
    vec = TfidfVectorizer().fit_transform(docs)
    sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    return int(sim * 100)

# --------------------------------------------
# ロング or ショート判定（ニュース内容ベース）
# --------------------------------------------
def decide_direction(gpt_lines, px_lines):
    text = " ".join(gpt_lines + px_lines).lower()
    bullish = ["rise", "gain", "reboun", "lower cpi", "rate cut", "risk-on", "上昇", "利下げ"]
    bearish = ["fall", "drop", "sell-off", "rate hike", "risk-off", "下落", "利上げ"]

    b = any(w in text for w in bullish)
    s = any(w in text for w in bearish)

    if b and not s:
        return "BUY"
    if s and not b:
        return "SELL"
    return "BUY"  # 同時判定時はBUY優先（あなたの方針）

# --------------------------------------------
# IFD-OCO (1ロット)
# --------------------------------------------
def calc_ifd(symbol, price, direction):
    tick_map = {"JP225": 5, "NAS100": 0.25, "GER40": 1, "XAUUSD": 0.05}
    atr_map  = {"JP225": 50, "NAS100": 40, "GER40": 35, "XAUUSD": 3}

    tick = tick_map[symbol]
    atr = atr_map[symbol]

    if direction == "BUY":
        entry = price + 3 * tick
        sl = price - atr
        tp1 = price + atr
        tp2 = price + atr * 1.5
    else:
        entry = price - 3 * tick
        sl = price + atr
        tp1 = price - atr
        tp2 = price - atr * 1.5

    return {
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "atr": atr
    }


# ============================================
# メイン：4銘柄一括処理
# ============================================
def swing_ai_batch(latest_prices: dict):
    """
    latest_prices = {"JP225": 50834.8, "NAS100": 25664.2, ...}
    """
    output = {}
    for symbol in SYMBOLS:
        price = latest_prices.get(symbol)
        if price is None:
            output[symbol] = {"error": "price missing"}
            continue

        g = fetch_gpt_news(symbol)
        p = fetch_px_news(symbol)
        cons = calc_consistency(g, p)
        dire = decide_direction(g, p)
        ifd = calc_ifd(symbol, price, dire)

        output[symbol] = {
            "news_gpt": g,
            "news_px": p,
            "consistency": cons,
            "direction": dire,
            "ifd": ifd
        }
    return output
