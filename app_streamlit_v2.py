import os
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

import streamlit as st
import json
import tempfile
import traceback
import pandas as pd
from vision_mobile_v13 import extract_watchlist_mobile_v13
from utils_price import build_ifdoco_lines
from decision_engine import (
    manual_table_rows,
    auto_table_rows,
    news_consensus_all,
)

# ファイル先頭あたりに追加
def _only_symbols(pr: dict) -> dict:
    SYM = {"JP225","NAS100","GER40","XAUUSD"}
    return {k: float(v) for k, v in pr.items() if k in SYM and isinstance(v, (int, float))}

def _md_table(rows: list[list[object]]) -> str:
    header = (
        "| trade_mode | 銘柄 | 方向 | entry_price | SL | TP1 | TP2 | order_type | 判定 | ニュースロック | 推奨度 | ロット | CUT条件 |\n"
        "|-------------|------|------|--------------|------|------|------|-------------|--------|----------------|----------|--------|-----------|\n"
    )
    def fmt(x):
        if isinstance(x, float) or isinstance(x, int):
            return f"{x:.1f}"
        return str(x)
    body = "\n".join("| " + " | ".join(fmt(c) for c in r) + " |" for r in rows)
    return header + body


# --- ニュース短観レンダリング（3–4行） -------------------------------
def _short(txt: str | None, limit: int = 120) -> str:
    s = (txt or "").strip().replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "…"

def _dir_jp(d: str | None) -> str:
    return "買い" if d == "BUY" else ("売り" if d == "SELL" else "中立")

def _news_digest_block(sym: str, stars: str, score: float, raw: dict) -> str:
    pplx = _short((raw or {}).get("pplx", {}).get("summary"))
    gpt  = _short((raw or {}).get("gpt",  {}).get("summary"))
    lines = [
        f"**{sym} — {stars}（score={score:.2f}） / 方向：{_dir_jp((raw or {}).get('direction'))}**",
        f"- PPLX: {pplx or '(PPLX要約なし)'}",
        f"- GPT : {gpt  or '(GPT要約なし)'}",
        "- 注: ニュースは方向性の補助。執行はIFDOCO条件を優先。"
    ]
    return "\n".join(lines)

def _render_news_digests_manual(meta: dict):
    st.subheader("📰 ニュース短観（各銘柄・3–4行）")
    summaries = (meta or {}).get("summaries", {}) or {}
    for sym in ("JP225","NAS100","GER40","XAUUSD"):
        s = summaries.get(sym)
        if not s:
            continue
        st.markdown(_news_digest_block(sym, s.get("stars","★☆☆☆☆"), s.get("score",0.0), s.get("raw", {})))
        st.markdown("---")

def _render_news_digests_auto(meta: dict, symbol_fallback: str | None = None):
    st.subheader("📰 ニュース短観（AI自動・3–4行）")
    raw   = (meta or {}).get("news", {}) or {}
    stars = (meta or {}).get("stars_txt", "★☆☆☆☆")
    score = float((meta or {}).get("score", 0.0) or 0.0)
    tvsym = ((meta or {}).get("tv") or {}).get("symbol") or symbol_fallback or "—"
    st.markdown(_news_digest_block(tvsym, stars, score, raw))

st.set_page_config(
    page_title="CFD3 Dual Engine",
    page_icon="🧠",
    layout="wide",
)

# 簡潔な全画面ダーク背景（優先度高）
st.markdown("""
<style>
html, body, [data-testid="stApp"] {
    background-color: #000000 !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# If background stays white, force text color to black while preserving the pink brain icon
st.markdown("""
<style>
/* Target most content text but exclude header brain */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    color: #000000 !important;
}

/* Make most inner elements' text black, but keep the brain icon and images */
[data-testid="stAppViewContainer"] *:not(.header-brain):not(.header-brain *) {
    color: #000000 !important;
}

/* Keep brain icon colors */
.header-brain, .header-brain .icon { color: #000000 !important; }
.header-brain .icon { background: transparent !important; color: #000000 !important; }

/* Links, buttons and code should be readable on white bg */
a, button, input, textarea { color: #000000 !important; }
pre, code { color: #000000 !important; background: transparent !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Force overall background to black */
:root, html, body, [data-testid="stAppViewContainer"], .main, .block-container, .reportview-container { background-color: #000000 !important; color: #e0e0e0 !important; }

/* Make header transparent and ensure no inherited background bleeds */
[data-testid="stHeader"], header, [data-testid="stHeader"] * { background: transparent !important; box-shadow: none !important; }

/* Sidebar and alert colors */
section[data-testid="stSidebar"] { background-color: #111111 !important; color: #e0e0e0 !important; }
.stAlert { background-color: #111111 !important; }

/* Code / pre on dark background */
pre, code { color: #e0e0e0 !important; background: transparent !important; }

/* Header brain: keep pink circle but constrain to its box */
.header-brain { display:inline-flex; align-items:center; gap:12px; font-size:28px; background: transparent !important; color: #000000 !important; }
.header-brain .icon { width:48px; height:48px; background: transparent !important; border-radius:0; display:inline-flex; align-items:center; justify-content:center; font-size:28px; box-shadow: none !important; color: #000000 !important; }
.header-brain .icon span { transform: translateY(-2px); color: #000000 !important; }

/* Prevent any element from using the pink as background for large areas */
*:not(.header-brain):not(.header-brain *) { background: transparent !important; }
</style>
<div class="header-brain"><div class="icon"><span>🧠</span></div><div>CFD3 Dual Engine — 手動 + AI自動 スイング1口 モード</div></div>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Strong enforcement: set page roots to black */
html, body { background-color: #000000 !important; }
[data-testid="stAppViewContainer"], .reportview-container, .main, .block-container, .stAppViewContainer, .stAppWindow-main { background-color: #000000 !important; }

/* Make all inner elements transparent so the black root shows through */
[data-testid="stAppViewContainer"] * { background: transparent !important; color: #e0e0e0 !important; }

/* Specific fallbacks for Streamlit-generated containers */
div[data-testid="stAppViewContainer"] > .main > .block-container, .stBlock, .element-container { background-color: #000000 !important; }

/* Preserve the brain icon (no pink background) */
.header-brain .icon { background: transparent !important; }

/* Ensure images and code blocks do not introduce white backgrounds */
img, .stImage, pre, code { background: transparent !important; }

</style>
""", unsafe_allow_html=True)
st.markdown("""
<script>
(function(){
    function preserveBrain(el){
        var icon = document.querySelector('.header-brain .icon');
        if(icon){
            try{ icon.style.setProperty('background-color', 'transparent', 'important'); icon.style.setProperty('background', 'transparent', 'important'); icon.style.setProperty('color', '#000000', 'important'); }catch(e){}
        }
    }

    function applyBlackTo(el){
        if(!el || (el.classList && (el.classList.contains('header-brain') || el.closest && el.closest('.header-brain')))) return;
        try{
            el.style.setProperty('background-color', '#000000', 'important');
            el.style.setProperty('background', 'transparent', 'important');
            el.style.setProperty('color', '#e0e0e0', 'important');
        }catch(e){}
    }

    function walkAndApply(root){
        if(!root) return;
        applyBlackTo(root);
        var nodes = root.querySelectorAll ? root.querySelectorAll('*') : [];
        for(var i=0;i<nodes.length;i++) applyBlackTo(nodes[i]);
    }

    function init(){
        try{
            document.documentElement.style.setProperty('background-color', '#000000', 'important');
            document.body.style.setProperty('background-color', '#000000', 'important');
            walkAndApply(document);
            preserveBrain();

            // Observe DOM changes and re-apply
            var mo = new MutationObserver(function(mutations){
                mutations.forEach(function(m){
                    m.addedNodes && m.addedNodes.forEach(function(node){
                        try{ walkAndApply(node); }catch(e){}
                    });
                });
                preserveBrain();
            });
            mo.observe(document.documentElement || document, { childList: true, subtree: true });

            // Safety re-apply a few times (for frameworks that update after load)
            var tries = 0; var t = setInterval(function(){ tries++; walkAndApply(document); preserveBrain(); if(tries>6) clearInterval(t); }, 500);
        }catch(e){ console.error(e); }
    }

    if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
</script>
""", unsafe_allow_html=True)

# Last-resort script: aggressively overwrite any background/gradient/opacity on all elements
st.markdown("""
<script>
(function(){
    function forceBlack(el){
        if(!el || (el.nodeType!==1)) return;
        if(el.closest && el.closest('.header-brain')) return; // keep header brain
        try{
            el.style.setProperty('background-image','none','important');
            el.style.setProperty('background','none','important');
            el.style.setProperty('background-color','#000000','important');
            el.style.setProperty('opacity','1','important');
            el.style.setProperty('color','#e0e0e0','important');
            el.style.setProperty('box-shadow','none','important');
            el.style.setProperty('filter','none','important');
        }catch(e){}
    }

    function applyAll(root){
        try{
            forceBlack(document.documentElement);
            forceBlack(document.body);
            var nodes = root.querySelectorAll ? root.querySelectorAll('*') : [];
            for(var i=0;i<nodes.length;i++){
                var n = nodes[i];
                // allow images and icons to render naturally
                if(n.tagName==='IMG' || n.tagName==='SVG' || n.classList.contains('stImage')) continue;
                forceBlack(n);
            }
        }catch(e){console.error(e)}
    }

    function init(){
        applyAll(document);
        var mo = new MutationObserver(function(muts){
            muts.forEach(function(m){
                m.addedNodes && m.addedNodes.forEach(function(n){ if(n.querySelectorAll) applyAll(n); });
            });
        });
        mo.observe(document.documentElement||document, { childList:true, subtree:true });
        // reapply a few times
        var t=0; var it=setInterval(function(){ applyAll(document); t++; if(t>8) clearInterval(it); }, 400);
    }
    if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
</script>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Aggressive fallback: force every element to black background and light text */
* , *::before, *::after { background-color: #000000 !important; background: #000000 !important; color: #e0e0e0 !important; }
/* Preserve the brain icon (transparent background, black text) */
.header-brain .icon { background-color: transparent !important; background: transparent !important; color: #000000 !important; }
/* Ensure images keep natural content but without white background */
img, .stImage { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# Additional injection: create a style element at runtime to override stubborn pseudo-elements and inline rules
st.markdown("""
<script>
(function(){
    try{
        var css = '* , *::before, *::after { background-color: #000000 !important; background: #000000 !important; color: #e0e0e0 !important; }\n.header-brain .icon { background: transparent !important; color: #000000 !important; }\nimg, .stImage { background: transparent !important; }';
        var s = document.createElement('style');
        s.type = 'text/css';
        s.appendChild(document.createTextNode(css));
        document.head.appendChild(s);
    }catch(e){console.error(e);}  
})();
</script>
""", unsafe_allow_html=True)
st.caption("高精度 OCR（v12）、ニュース短観、手動とAI自動の2モード搭載 完全版")

uploaded = st.file_uploader("GMOアプリのスクリーンショットをアップロード", type=["png","jpg","jpeg"])

prices: dict = {}
if uploaded:
    tmp_path = "tmp_upload.png"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getvalue())

    st.caption("OCR を実行中…（v13: EasyOCR→Tesseract ラベル連動）")
    try:
        prices = extract_watchlist_mobile_v13(tmp_path)
        st.success("読み取り成功（mobile-v13-easyocr+tesseract-labeled）")
        st.code(json.dumps(prices, ensure_ascii=False, indent=2))
    except Exception as e:
        st.error(f"OCRエラー: {e}")
        prices = {}
else:
    st.warning("スクショをアップロードしてください。")

if not prices:
    st.stop()

# 2. ニュース短観（全銘柄）
st.header("📰 全銘柄ニュース短観（GPT × Perplexity）")
news_all = news_consensus_all()
for sym, d in news_all.items():
    st.markdown(f"### {sym} — {d['stars']}（score={d['score']:.2f}）")
    st.write("**方向**：", d["direction"])
    st.write("**PPLX**：", d["pplx"]["summary"])
    st.write("**GPT**：", d["gpt"]["summary"])
    st.markdown("---")

# 3. モード選択
st.subheader("📡 最終判定（片側 only：必ず表を出力）")
mode = st.radio("モードを選択", ["手動（★推奨・片側1本）", "AI自動（スイング1口・慎重型）"], index=0, horizontal=True)

if mode.startswith("手動"):
    if st.button("🔎 手動：ニュース反映で4銘柄ぶんを作成", disabled=not prices):
        rows, meta = manual_table_rows(prices, lots=1, trade_mode="DAY6H")
        st.markdown(_md_table(rows))
        try:
            _render_news_digests_manual(meta)
        except Exception:
            pass
        with st.expander("📰 手動モード ニュース短観（各銘柄）", expanded=False):
            for sym, s in (meta.get("summaries") or {}).items():
                st.markdown(f"**{sym}** — {s['stars']} (score={s['score']:.2f}) / dir={s['dir']}")
                raw = s["raw"]
                st.write("PPLX:", raw.get("pplx",{}).get("summary","(none)"))
                st.write("GPT :", raw.get("gpt" ,{}).get("summary","(none)"))
                st.markdown("---")
else:
    gate = st.slider("参戦ゲート（AI自動：一致スコア）", 0.60, 0.99, 0.90, 0.01)
    if st.button("✅ 自動：TV×ニュース一致で判定（常に1行は出力）", disabled=not prices):
        rows, meta = auto_table_rows(prices, gate=gate, lots=1, trade_mode="SWING1")
        st.markdown(_md_table(rows))
        st.caption(f"一致スコア: {meta.get('score',0):.2f} / ★: {meta.get('stars_txt','-')} / 判定: {meta.get('verdict','-')}（閾値={int(gate*100)}%）")
        try:
            sym_fb = None
            if rows and len(rows) > 0:
                sym_fb = rows[0][1]
            _render_news_digests_auto(meta, symbol_fallback=sym_fb)
        except Exception:
            pass

    # Ensure `meta` exists to avoid NameError when no auto run occurred
    meta = locals().get('meta', {})

    st.markdown(f"""
### 判定結果
- **方向**：{meta.get("direction")}
- **スコア**：{meta.get("score"):.2f}
- **推奨度**：{meta.get("stars")}
- **最終判定**：{meta.get("verdict")}
""")

    with st.expander("📰 AI自動モード ニュース短観（使用した銘柄）"):
        st.json(meta.get("news"))

    if not rows:
        st.warning("一致スコアが基準を満たさないため、今回は “見送り” です。")
        st.stop()

    df = pd.DataFrame(rows, columns=[
        "trade_mode","銘柄","方向","entry_price","SL","TP1","TP2",
        "order_type","判定","ニュースロック","推奨度","ロット","CUT条件"
    ])
    st.dataframe(df, width="stretch")

    st.subheader("🧾 GMO 用 IFDOCO（コピペ可）")
    st.code(build_ifdoco_lines(rows), language="text")
