#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""語尾の出る順と発音のずれの集計(GOSE・BTS) — 段階実行

  python3 gobi.py survey --gose seventeen_analysis/output/ani_textbook_pipeline/corpus.csv \
      --bts bts_pipeline/bts_textbook_pipeline/bts_corpus.csv
      → gobi_analysis/step0/ (STEP0 ファイルの場所・行数・除外数。kiwi は使わない)
  python3 gobi.py draft --gose ... --bts ...
      → gobi_analysis/step1_2/ (STEP1 抽出・言い回し候補、STEP2 統合表の案。集計はしない)

前処理V2([..]・(..) を消す、空白の正規化、ハングルのある行だけ)。1行=1発話、行末で判定。
既存の出力は上書きしない(出力フォルダに既にファイルがあれば停止)。
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

BR = re.compile(r"\[[^\]]*\]")
PA = re.compile(r"\([^)]*\)")
HANGUL = re.compile(r"[가-힣]")
TELOP_TAG = "0_対象外_텔롭"
JAMO = str.maketrans({"ᆫ": "ㄴ", "ᆯ": "ㄹ", "ᆸ": "ㅂ", "ᆷ": "ㅁ", "ᆻ": "ㅆ"})


def clean(t):
    return re.sub(r"\s+", " ", PA.sub(" ", BR.sub(" ", t))).strip()


def nf(f):
    return f.translate(JAMO)


def bt(tag):
    return tag.split("-")[0]


def outdir(p):
    if os.path.exists(p) and os.listdir(p):
        sys.exit(f"{p} に既にファイルがあります。上書きしないため停止します。")
    os.makedirs(p, exist_ok=True)
    return p


def wcsv(path, head, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


def is_telop(row):
    """(タグ一致, channel=텔롭, [..]だけの行)"""
    tag = any(v and TELOP_TAG in str(v) for v in row.values() if isinstance(v, str))
    ch = any(k and "channel" in k.lower() and (row.get(k) or "").strip() == "텔롭" for k in row if isinstance(k, str))
    t = row.get("text") or ""
    heur = bool(BR.search(t)) and not BR.sub("", t).strip()
    return tag, ch, heur


def load_gose(path):
    """発話行 [(話, 行番号, 本文)] と除外の内訳"""
    st = Counter()
    eps = set()
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        for i, r in enumerate(rd, 1):
            st["全行"] += 1
            eps.add(r.get("episode", ""))
            tag, ch, heur = is_telop(r)
            if tag or ch or heur:
                st["テロップ:タグ一致" if tag else "テロップ:channel=텔롭" if ch else "テロップ:[..]だけの行"] += 1
                continue
            raw = r.get("text") or ""
            if PA.search(raw):
                st["参考:(..)を含む行(消して残す)"] += 1
            if BR.search(raw):
                st["参考:行内の[..]を含む行(消して残す)"] += 1
            t = clean(raw)
            if not HANGUL.search(t):
                st["ハングルなし"] += 1
                continue
            out.append((r.get("episode", ""), i, t))
    st["発話行"] = len(out)
    return cols, st, len(eps), out


def load_bts(path):
    st = defaultdict(Counter)
    clips = defaultdict(set)
    out = defaultdict(list)
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        for r in rd:
            s = r.get("source", "")
            st[s]["全行"] += 1
            clips[s].add(r.get("clip_id", ""))
            tag, ch, heur = is_telop(r)
            if tag or ch or heur:
                st[s]["テロップ:タグ一致" if tag else "テロップ:channel=텔롭" if ch else "テロップ:[..]だけの行"] += 1
                continue
            t = clean(r.get("text") or "")
            if not HANGUL.search(t):
                st[s]["ハングルなし"] += 1
                continue
            out[s].append((r.get("clip_id", ""), r.get("line_no", ""), t))
    for s in st:
        st[s]["発話行"] = len(out[s])
    return cols, st, {s: len(v) for s, v in clips.items()}, out


# ------------------------------------------------------------------ STEP0
def survey(a):
    o = outdir(os.path.join(a.out, "step0"))
    R = ["# STEP0 ファイルの場所・行数・除外数\n"]
    R.append(f"- 探したフォルダ: {os.path.abspath(a.root)}\n\n## text 列を持つ CSV\n")
    R.append("| ファイル | 行数 | 列 | channel列 | 「텔롭」を含む行 | 「0_対象外_텔롭」を含む行 |\n|---|---:|---|---|---:|---:|")
    for dp, dn, fn in os.walk(a.root):
        dn[:] = [d for d in dn if not d.startswith(".") and d != "gobi_analysis"]
        for n in sorted(fn):
            if not n.endswith(".csv"):
                continue
            p = os.path.join(dp, n)
            try:
                if os.path.getsize(p) > 500_000_000:
                    continue
                with open(p, encoding="utf-8-sig", newline="", errors="replace") as f:
                    rd = csv.DictReader(f)
                    cols = rd.fieldnames or []
                    if "text" not in cols:
                        continue
                    n_rows = tel = tag = 0
                    for r in rd:
                        n_rows += 1
                        vals = " ".join(str(v) for v in r.values() if isinstance(v, str))
                        tel += "텔롭" in vals
                        tag += TELOP_TAG in vals
            except Exception as e:
                R.append(f"| {p} | 読めない({type(e).__name__}) | | | | |")
                continue
            chc = [c for c in cols if c and "channel" in c.lower()]
            R.append(f"| {p} | {n_rows} | {', '.join(c for c in cols if c)} | {', '.join(chc) or 'なし'} | {tel} | {tag} |")

    cols, st, n_ep, _ = load_gose(a.gose)
    R.append(f"\n## GOSE: {a.gose}\n\n- 列: {cols}\n- 話数: {n_ep}")
    for k in ("全行", "テロップ:タグ一致", "テロップ:channel=텔롭", "テロップ:[..]だけの行", "ハングルなし", "発話行",
              "参考:(..)を含む行(消して残す)", "参考:行内の[..]を含む行(消して残す)"):
        R.append(f"- {k}: {st[k]}")
    cols, st, clips, _ = load_bts(a.bts)
    R.append(f"\n## BTS: {a.bts}\n\n- 列: {cols}\n\n| source | クリップ数 | 全行 | テロップ:タグ一致 | テロップ:channel | テロップ:[..]だけ | ハングルなし | 発話行 |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in sorted(st):
        c = st[s]
        R.append(f"| {s} | {clips[s]} | {c['全行']} | {c['テロップ:タグ一致']} | {c['テロップ:channel=텔롭']} | "
                 f"{c['テロップ:[..]だけの行']} | {c['ハングルなし']} | {c['発話行']} |")
    txt = "\n".join(R)
    open(os.path.join(o, "step0_files.md"), "w", encoding="utf-8").write(txt)
    print(txt)


# ------------------------------------------------------------------ STEP0 補足:テロップの照合
def nk(s):
    return re.sub(r"[^가-힣A-Za-z0-9]", "", s or "")


def telop_check(a):
    import random
    o = outdir(os.path.join(a.out, "step0_telop"))
    rows = []
    with open(a.gose, encoding="utf-8-sig", newline="") as f:
        for i, r in enumerate(csv.DictReader(f), 1):
            raw = r.get("text") or ""
            tag, ch, heur = is_telop(r)
            if tag or ch or heur:
                stt = "[..]だけの行として除外"
            elif not HANGUL.search(clean(raw)):
                stt = "ハングルなしで除外"
            else:
                stt = "発話行に残る"
            rows.append((i, r.get("episode", ""), raw, stt))
    by_ep_key = defaultdict(list)
    by_key = defaultdict(list)
    by_ep = defaultdict(list)
    for k, (i, ep, raw, stt) in enumerate(rows):
        for key in {nk(raw), nk(clean(raw))}:
            if key:
                by_ep_key[(ep, key)].append(k)
                by_key[key].append(k)
        by_ep[ep].append(k)
    base = os.path.dirname(os.path.abspath(a.gose))
    files = []
    chv = Counter()
    kw = {}
    for dp, dn, fn in os.walk(base):
        for n in sorted(fn):
            if not n.endswith(".csv") or os.path.abspath(os.path.join(dp, n)) == os.path.abspath(a.gose):
                continue
            p = os.path.join(dp, n)
            with open(p, encoding="utf-8-sig", newline="", errors="replace") as f:
                rd = csv.DictReader(f)
                chc = [c for c in (rd.fieldnames or []) if c and "channel" in c.lower()]
                if not chc or "text" not in (rd.fieldnames or []):
                    continue
                files.append(os.path.relpath(p, base))
                for r in rd:
                    v = (r.get(chc[0]) or "").strip()
                    chv[v] += 1
                    if "텔롭" not in v:
                        continue
                    t = r.get("raw_text") or r.get("text") or ""
                    key = (r.get("episode", ""), nk(t) or nk(r.get("text")))
                    if key[1] and key not in kw:
                        kw[key] = (os.path.relpath(p, base), t, r.get("text") or "")
    eps = set(by_ep)
    res = []
    for (ep, key), (src, t, t2) in kw.items():
        hit, how = by_ep_key.get((ep, key)), "完全一致(同じ話)"
        if not hit and nk(t2) and nk(t2) != key:
            hit = by_ep_key.get((ep, nk(t2)))
        if not hit and ep not in eps:
            hit, how = by_key.get(key), "完全一致(話の列が合わず本文だけ)"
        if not hit:
            hit, how = [k for k in by_ep.get(ep, []) if key in nk(rows[k][2])], "部分一致(用例の本文が行の一部)"
        if not hit:
            res.append([src, ep, t, "一致なし", "", "", ""])
            continue
        sts = sorted({rows[k][3] for k in hit})
        stt = "発話行に残る" if "発話行に残る" in sts else sts[0]
        res.append([src, ep, t, stt, how + ("(複数行)" if len(hit) > 1 else ""), ";".join(str(rows[k][0]) for k in hit[:5]),
                    rows[hit[0]][2]])
    wcsv(os.path.join(o, "telop_check.csv"), ["用例ファイル(最初に見つかったもの)", "話", "用例の本文", "corpus.csv での扱い", "一致の仕方",
                                               "corpus.csv の行番号(最大5)", "corpus.csv の本文(最初の行)"], res)
    R = ["# STEP0 補足 テロップの照合\n", f"- 対象: {base} 以下で channel 列を持つ CSV {len(files)} 件",
         f"- channel=텔롭 の用例(話+本文で重複を除く): {len(kw)} 件",
         "- channel 列の値の内訳(全ファイル・重複込み): " + "、".join(f"{k or '(空欄)'} {v}" for k, v in chv.most_common(10)),
         "\n| corpus.csv での扱い | 件数 | 割合(分母: 用例 %d) |\n|---|---:|---:|" % len(kw)]
    c = Counter(r[3] for r in res)
    for k, v in c.most_common():
        R.append(f"| {k} | {v} | {v / max(len(kw), 1) * 100:.1f}% |")
    R.append("\n| 一致の仕方 | 件数 |\n|---|---:|")
    for k, v in Counter(r[4].split("(複数行)")[0] or "一致なし" for r in res).most_common():
        R.append(f"| {k} | {v} |")
    left = [r for r in res if r[3] == "発話行に残る"]
    R.append(f"\n## 発話行に残っている用例 {len(left)} 件のうち10行(シード42)\n\n| 話 | 用例の本文 | corpus.csv の本文 | 一致の仕方 |\n|---|---|---|---|")
    for r in random.Random(42).sample(left, min(10, len(left))):
        R.append(f"| {r[1]} | {r[2][:60]} | {r[6][:60]} | {r[4]} |")
    R.append("\n全件: telop_check.csv")
    txt = "\n".join(R)
    open(os.path.join(o, "telop_check.md"), "w", encoding="utf-8").write(txt)
    print(txt)


# ------------------------------------------------------------------ STEP1・2
PRED = {"VV", "VA", "VX", "VCP", "VCN", "XSV", "XSA"}
NOUNISH = {"NNG", "NNP", "NNB", "NR", "NP", "MAG", "MAJ"}
NO_STRIP = {"세요", "에요", "예요"}
# 으・은・을・읍・는 の有無(まとめ方 final と同じ)
F_EF = {"을까": "ㄹ까", "을게": "ㄹ게", "을래": "ㄹ래", "으세요": "세요", "으세": "세요", "은가": "ㄴ가", "는가": "ㄴ가",
        "은데": "ㄴ데", "는데": "ㄴ데", "는다": "ㄴ다", "으냐": "냐", "느냐": "냐", "ㅂ니다": "습니다", "ㅂ니까": "습니까",
        "읍시다": "ㅂ시다", "으니": "니", "으라": "라"}
F_EC = {"으면": "면", "은데": "ㄴ데", "는데": "ㄴ데", "으니까": "니까", "으러": "러", "으려고": "려고", "을게": "ㄹ게"}
# まとめ方 final2 で足す規則(final の項目 → final2 の項目)
FINAL2 = {"야": ("어", "final2:어・아・야→어"), "ㄴ다": ("다", "final2:다・ㄴ다→다"), "ㄴ가": ("나", "final2:나・ㄴ가→나"),
          "ㄴ다고": ("다고", "final2:다고・ㄴ다고→다고"), "는다고": ("다고", "final2:다고・ㄴ다고→다고(는다고を含む)")}
ETM_N = {"을": "ㄹ", "ㄹ": "ㄹ", "은": "ㄴ", "ㄴ": "ㄴ", "는": "는", "던": "던"}
P0_FINAL = {"EF行": 50697, "final上位14": 45249}
EXPECT_FINAL2_TOP12 = 90.1


def norm_ending(tag, form):
    """final の項目と、適用した規則のリスト"""
    rules = []
    f = nf(form)
    if f == "죠":
        f = "지"
        rules.append("縮約(죠=지요)")
    elif f.endswith("요") and len(f) > 1 and f not in NO_STRIP:
        f = f[:-1]
        rules.append("요の有無")
    if f[0] in "아여":
        g = "어" + f[1:]
        if g != f:
            f = g
            rules.append("아/어/여")
    m = (F_EF if tag == "EF" else F_EC).get(f)
    if m:
        f = m
        rules.append("으・는などの有無")
    return f, rules


def phrase_of(core, j):
    """core[j] = 文末の用言(EPの手前)。語をまたぐ言い回しなら (候補キー, 開始位置, 推奨) を返す"""
    p = core[j]
    pt, pf = bt(p.tag), nf(p.form)
    a1 = core[j - 1] if j >= 1 else None
    a2 = core[j - 2] if j >= 2 else None
    if a1 is not None and a2 is not None and bt(a1.tag) == "NNB" and bt(a2.tag) == "ETM":
        etm = ETM_N.get(nf(a2.form), nf(a2.form))
        nb = "거" if nf(a1.form) in ("거", "것") else nf(a1.form)
        if nb == "거" and pt == "VCP":
            return f"-{etm} 거(이다)", j - 2, True
        if nb == "거" and pt == "VA" and pf == "같":
            return f"-{etm} 거 같다", j - 2, True
        if nb == "수" and pf in ("있", "없"):
            return f"-{etm} 수 {pf}다", j - 2, True
        if nb == "줄" and pf in ("알", "모르"):
            return f"-{etm} 줄 {pf}다", j - 2, False
        return f"-{etm} {nb}+{pf}/{pt}", j - 2, False
    if a1 is not None and bt(a1.tag) == "EC" and pt in ("VX", "VV", "VA"):
        ec, _ = norm_ending("EC", a1.form)
        if pt == "VX" or (ec in ("어야", "어도", "게") and pf == "되"):
            return f"-{ec} {pf}다", j - 1, False
    return None, None, False


def extract(kiwi, items, corpus):
    rows = []
    for (ep, row, t), toks in zip(items, kiwi.tokenize([x[2] for x in items])):
        core = [x for x in toks if not x.tag.startswith("S")]
        d = {"corpus": corpus, "ep": ep, "row": row, "text": t, "an": " ".join(f"{x.form}/{x.tag}" for x in core)}
        rows.append(d)
        if not core:
            d["kind"] = "除外:形態素なし"
            continue
        end = len(core) - 1
        yo_tail = None
        if end > 0 and core[end].form == "요" and core[end].tag in ("JX", "MM"):
            yo_tail = core[end]
            end -= 1
        e = core[end]
        tg = bt(e.tag)
        if tg == "EF" and nf(e.form) == "요":
            pt = bt(core[end - 1].tag) if end >= 1 else "(なし)"
            if pt in ("EF", "EC"):
                yo_tail = e          # 単独の 요(EF) は直前の語尾の 요(まとめ方 final と同じ)
                end -= 1
                e = core[end]
                tg = pt
            elif pt.startswith("J") or pt == "VCP":
                d["kind"] = "除外:語尾なし(単独の요・直前が助詞/指定詞。final2)"
                continue
            elif pt in NOUNISH:
                d["kind"] = "除外:語尾なし(単独の요・直前が名詞類)"
                continue
            else:
                d["kind"] = f"除外:未区分(単独の요・直前が{pt})"
                continue
        if tg not in ("EF", "EC"):
            d["kind"] = f"除外:その他の語尾({tg})" if tg.startswith("E") else f"除外:語尾なし({tg})"
            continue
        j = end - 1
        eps = []
        while j >= 0 and bt(core[j].tag) == "EP":
            eps.insert(0, nf(core[j].form))
            j -= 1
        item, rules = norm_ending(tg, e.form)
        if yo_tail is not None and "요の有無" not in rules:
            rules.insert(0, "요の有無")
        if item in ("에요", "예요"):
            if j >= 0 and bt(core[j].tag) in ("VCP", "VCN"):
                item = "야"
                rules.append("에요/예요→야(指定詞の後)")
            else:
                rules.append("要確認:에요/예요が指定詞の後でない")
        item2 = item
        if item in FINAL2:
            item2, r2 = FINAL2[item]
            rules.append(r2)
        d.update(kind="語尾", tag=tg, surf=nf(e.form) + ("+요" if yo_tail is not None else ""), item=item, item2=item2,
                 rules="・".join(rules), ep_list="+".join(eps), pred="", pred_ok=True, phrase="", phrase_rec=False,
                 ph_start="")
        if j >= 0 and bt(core[j].tag) in PRED:
            d["pred"] = f"{nf(core[j].form)}/{core[j].tag}"
            ph, st_i, rec = phrase_of(core, j)
            if ph:
                d["phrase"], d["phrase_rec"], d["ph_start"] = ph, rec, core[st_i].start
        else:
            d["pred_ok"] = False
            d["pred"] = f"{core[j].form}/{core[j].tag}" if j >= 0 else "(なし)"
        last = yo_tail if yo_tail is not None else e
        d["e_start"], d["e_end"] = e.start, last.start + last.len
        d["spell"] = t[e.start:last.start + last.len]
        d["phon"] = phon(t, core, end, j, d["e_end"], None)
        if d["phrase"]:
            d["phon_ph"] = phon(t, core, end, j, d["e_end"], st_i)
    return rows


def ef_table(rs, key):
    c = Counter(d[key] for d in rs if d["kind"] == "語尾" and d["tag"] == "EF")
    return c, sum(c.values())


def draft(a):
    import kiwipiepy
    from kiwipiepy import Kiwi
    o = outdir(os.path.join(a.out, "step1_2"))
    kiwi = Kiwi()
    _, gst, _, g = load_gose(a.gose)
    _, bst, _, b = load_bts(a.bts)
    data = {"GOSE": extract(kiwi, g, "GOSE"), "run": extract(kiwi, b.get("run", []), "run"),
            "bomb": extract(kiwi, b.get("bomb", []), "bomb")}
    R = [f"# STEP1・2 抽出と統合表の案(まとめ方 final2。集計前・要確認)\n\nkiwipiepy {kiwipiepy.__version__} / 前処理V2 / 1行=1発話、行末で判定\n"]

    head = ["corpus", "話", "行番号", "行", "区分", "タグ", "表層形", "項目(final)", "項目(final2)", "規則", "先語末語尾",
            "直前の用言", "直前が用言", "言い回し候補", "言い回し推奨", "語尾開始位置", "語尾終了位置", "言い回し開始位置", "綴り", "解析"]
    for n, rs in data.items():
        wcsv(os.path.join(o, f"extract_{n}.csv"), head,
             [[d["corpus"], d["ep"], d["row"], d["text"], d["kind"], d.get("tag", ""), d.get("surf", ""), d.get("item", ""),
               d.get("item2", ""), d.get("rules", ""), d.get("ep_list", ""), d.get("pred", ""),
               int(d["pred_ok"]) if d["kind"] == "語尾" else "", d.get("phrase", ""),
               int(d["phrase_rec"]) if d.get("phrase") else "", d.get("e_start", ""), d.get("e_end", ""),
               d.get("ph_start", ""), d.get("spell", ""), d["an"]] for d in rs])

    # 再現確認(行末がEFの行だけで、まとめ方 final の集計と比べる)
    cf, nef = ef_table(data["GOSE"], "item")
    c2, _ = ef_table(data["GOSE"], "item2")
    top14 = sum(v for _, v in cf.most_common(14))
    top12 = sum(v for _, v in c2.most_common(12))
    R.append("## 再現確認(GOSE、行末がEFの行。分母は行末がEFの行)\n\n| 項目 | 期待値 | 今回 | 一致 |\n|---|---:|---:|---|")
    R.append(f"| EF行 | {P0_FINAL['EF行']} | {nef} | {'○' if nef == P0_FINAL['EF行'] else '×'} |")
    R.append(f"| final 上位14の合計 | {P0_FINAL['final上位14']} | {top14} | {'○' if top14 == P0_FINAL['final上位14'] else '×'} |")
    p12 = top12 / nef * 100 if nef else 0
    R.append(f"| final2 上位12の占有率 | {EXPECT_FINAL2_TOP12}% | {top12}/{nef} = {p12:.1f}% | "
             f"{'○' if round(p12, 1) == EXPECT_FINAL2_TOP12 else '×'} |")
    R.append("\n### final と final2 の上位14(GOSE、行末がEFの行、分母 %d)\n\n| 順位 | final | 件数 | 累積 | final2 | 件数 | 累積 | 変化 |\n|---:|---|---:|---:|---|---:|---:|---|" % nef)
    l1, l2 = cf.most_common(14), c2.most_common(14)
    s1 = s2 = 0
    set1, set2 = {k for k, _ in l1}, {k for k, _ in l2}
    for i in range(14):
        k1, v1 = l1[i] if i < len(l1) else ("", 0)
        k2, v2 = l2[i] if i < len(l2) else ("", 0)
        s1 += v1
        s2 += v2
        ch = "final2で入った" if k2 and k2 not in set1 else ("順位が変わった" if k1 != k2 else "")
        R.append(f"| {i + 1} | {k1} | {v1} | {s1 / nef * 100:.1f}% | {k2} | {v2} | {s2 / nef * 100:.1f}% | {ch} |")
    R.append(f"\n- final の上位14にあって final2 の上位14にない: {sorted(set1 - set2)}(final2 でまとめられた項目を含む)")
    R.append(f"- final2 の上位14にあって final の上位14にない: {sorted(set2 - set1)}")

    # 行の内訳
    R.append("\n## STEP1 行末の内訳(分母:発話行)\n\n| 区分 | GOSE | Run BTS | BOMB |\n|---|---:|---:|---:|")
    kc = {n: Counter(d["kind"] for d in rs) for n, rs in data.items()}
    kinds = sorted(set().union(*kc.values()), key=lambda k: (k != "語尾", -kc["GOSE"][k]))
    for k in kinds:
        R.append(f"| {k} | " + " | ".join(f"{kc[n][k]} ({kc[n][k] / max(len(data[n]), 1) * 100:.1f}%)" for n in data) + " |")
    R.append("| 発話行(分母) | " + " | ".join(str(len(data[n])) for n in data) + " |")
    npo = {n: sum(1 for d in rs if d["kind"] == "語尾" and not d["pred_ok"]) for n, rs in data.items()}
    R.append("\n- 語尾の直前が用言・先語末語尾でない行(kiwiの誤りの疑い。「語尾」に含めたまま。要確認): "
             + " / ".join(f"{n} {npo[n]}" for n in data))

    # 言い回し候補
    pc = defaultdict(lambda: {"GOSE": 0, "run": 0, "bomb": 0, "rec": False, "ex": "", "ends": Counter()})
    for n, rs in data.items():
        for d in rs:
            if d.get("phrase"):
                x = pc[d["phrase"]]
                x[n] += 1
                x["rec"] = d["phrase_rec"]
                x["ends"][d["item2"]] += 1
                if not x["ex"]:
                    x["ex"] = d["text"]
    ph_rows = sorted(pc.items(), key=lambda kv: -(kv[1]["GOSE"] + kv[1]["run"] + kv[1]["bomb"]))
    ph5 = [(k, v) for k, v in ph_rows if v["GOSE"] + v["run"] + v["bomb"] >= 5]
    wcsv(os.path.join(o, "phrase_candidates.csv"),
         ["言い回し", "GOSE", "Run BTS", "BOMB", "推奨(○=1項目として拾う案)", "後ろの語尾の内訳(final2、上位5)", "例", "採用(○/×。空欄=未確認)"],
         [[k, v["GOSE"], v["run"], v["bomb"], "○" if v["rec"] else "要確認",
           "、".join(f"{e}{c}" for e, c in v["ends"].most_common(5)), v["ex"], ""] for k, v in ph5])
    R.append(f"\n## STEP1 語をまたぐ言い回しの候補 → phrase_candidates.csv(合計5件以上 {len(ph5)}種類)\n")
    R.append("| 言い回し | GOSE | Run BTS | BOMB | 推奨 |\n|---|---:|---:|---:|---|")
    for k, v in ph_rows[:25]:
        R.append(f"| {k} | {v['GOSE']} | {v['run']} | {v['bomb']} | {'○' if v['rec'] else '要確認'} |")

    # 統合表
    mt = defaultdict(lambda: {"GOSE": 0, "run": 0, "bomb": 0, "npo": 0, "spell": Counter(), "ex": "", "rules": "", "item": ""})
    item_tags = defaultdict(Counter)
    for n, rs in data.items():
        for d in rs:
            if d["kind"] != "語尾":
                continue
            x = mt[(d["tag"], d["surf"], d["item2"])]
            x[n] += 1
            x["npo"] += not d["pred_ok"]
            x["spell"][d["spell"]] += 1
            x["rules"], x["item"] = d["rules"], d["item"]
            if not x["ex"]:
                x["ex"] = d["text"]
            item_tags[d["item2"]][d["tag"]] += 1
    rows = []
    for (tg, sf, it), v in sorted(mt.items(), key=lambda kv: -(kv[1]["GOSE"] + kv[1]["run"] + kv[1]["bomb"])):
        tot = v["GOSE"] + v["run"] + v["bomb"]
        why = []
        if "要確認" in v["rules"]:
            why.append(v["rules"].split("要確認:")[1].split("・")[0])
        if "는다고を含む" in v["rules"]:
            why.append("는다고 を 다고 に入れた")
        if it in NO_STRIP:
            why.append("요を外さない形")
        tg_c = item_tags[it]
        if len(tg_c) > 1 and min(tg_c.values()) / sum(tg_c.values()) >= 0.1:
            why.append(f"同じ項目にEFとECが混在(EF{tg_c['EF']}/EC{tg_c['EC']})")
        if v["npo"] / tot >= 0.2:
            why.append(f"直前が用言でない {v['npo']}件")
        if it in ("잖아", "대", "래", "냬", "재"):
            why.append("縮約形(-지 않아/-다고 해 など)と別項目のままにしている")
        rows.append([sf, tg, v["item"], it, v["rules"], v["GOSE"], v["run"] + v["bomb"], v["run"], v["bomb"],
                     "、".join(f"{s}{c}" for s, c in v["spell"].most_common(3)), v["ex"], "要確認:" + "/".join(why) if why else ""])
    wcsv(os.path.join(o, "merge_table.csv"),
         ["表層形(kiwi)", "タグ", "項目(final)", "項目(final2)", "適用した規則", "GOSE", "BTS計", "Run BTS", "BOMB",
          "字幕上の綴り(上位3)", "例", "要確認"], rows)
    flagged = sum(1 for r in rows if r[-1])
    R.append(f"\n## STEP2 統合表の案 → merge_table.csv\n\n- 表層形 {len(rows)} 行 → 項目(final2) {len({r[3] for r in rows})} 個 / 要確認の印 {flagged} 行")
    R.append("- 規則: 요の有無・죠=지요・아/어/여・으/는などの有無(final と同じ表)・에요/예요→야(指定詞の後)、"
             "final2: 어・아・야→어、다・ㄴ다→다、나・ㄴ가→나、다고・ㄴ다고→다고、単独の요(直前が助詞・指定詞)→語尾なし。습니다と습니까は別項目")
    R.append("- EF と EC は同じ項目にまとめている(行末の EC を含む)。STEP3 の分母はこの「語尾」の行")
    R.append("- STEP5 の注記: 判定の範囲は「語幹の最後の音節＋語尾全体」。名詞・副詞＋하다(생각해[생가캐]・못해[모태])の激音化は"
             "名詞・副詞と하の境目で起きるため、範囲の外になり数えない")
    R.append("\n### 項目(final2)の上位30(EF+EC、言い回しを当てる前。GOSE件数順)\n\n| 項目 | GOSE | BTS計 | 表層形の数 |\n|---|---:|---:|---:|")
    agg = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        agg[r[3]][0] += r[5]
        agg[r[3]][1] += r[6]
        agg[r[3]][2] += 1
    for it, (gc, bc, ns) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:30]:
        R.append(f"| {it} | {gc} | {bc} | {ns} |")
    R.append("\n### 要確認の行(合計件数の上位20)\n\n| 表層形 | タグ | 項目(final2) | GOSE | BTS計 | 理由 |\n|---|---|---|---:|---:|---|")
    for r in [r for r in rows if r[-1]][:20]:
        R.append(f"| {r[0]} | {r[1]} | {r[3]} | {r[5]} | {r[6]} | {r[-1][4:]} |")
    R.append("\n確認してほしいもの: phrase_candidates.csv の採用欄、merge_table.csv の項目と要確認の行。確認が済むまで STEP3 以降は実行しない。")
    txt = "\n".join(R)
    open(os.path.join(o, "step1_2_report.md"), "w", encoding="utf-8").write(txt)
    print(txt)


# ------------------------------------------------------------------ STEP5 発音のずれ(判定のみ。集計は STEP3 の後)
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
        "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
OBS = set("ㄱㄲㅋㄳㄺ" "ㄷㅅㅆㅈㅊㅌ" "ㅂㅍㄿㅄ")      # ㄱㄷㅂ系(ㅆは子音の前で ㄷ系)
HJ = set("ㅎㄶㅀ")
LEVEL = {"変化なし": "低", "有声音化": "低", "パッチムの後の濃音化": "低", "ㄹの後の濃音化": "中", "連音化": "高",
         "ㅎの脱落＋連音化": "高", "鼻音化": "高", "激音化": "高",
         "激音化・ㅎ＋ㅅ": "高"}
CJAMO = set("ᆫᆯᆸᆷᆻ")


def sy(ch):
    o = ord(ch) - 0xAC00
    return (CHO[o // 588], JONG[o % 28]) if 0 <= o < 11172 else None


def pair_type(a, b, src):
    """a,b = 音節。src = a のパッチムの出どころ('語尾ㄹ' / '語幹' / '語尾' )。種類の名前を返す(None=変化なし)"""
    ja, cb = sy(a)[1], sy(b)[0]
    if ja == "":
        return "有声音化" if cb in "ㄱㄷㅂㅈ" else None
    if cb == "ㅇ":
        if ja in HJ:
            return "ㅎの脱落＋連音化"
        return None if ja == "ㅇ" else "連音化"
    if ja in HJ:
        if cb in "ㄱㄷㅈ":
            return "激音化"                                          # 좋다[조타]・그렇지[그러치]
        if cb == "ㅅ":
            return "激音化・ㅎ＋ㅅ"                                   # 좋습니다[조씀니다](濃音化として記録)
        if cb == "ㄴ":
            return "表にない:ㅎ系+ㄴ"
        return None
    if ja in OBS:
        if cb in "ㄱㄷㅂㅅㅈ":
            return "パッチムの後の濃音化"
        if cb in "ㄴㅁ":
            return "鼻音化"
        if cb == "ㄹ":
            return "表にない:鼻音化(ㄹ→ㄴ)"
        return None
    if ja == "ㄹ" and src == "語尾ㄹ" and cb in "ㄱㄷㅂㅅㅈ":
        return "ㄹの後の濃音化"                                   # 標準発音法 第27項・付記(有声音化より優先)
    if (ja, cb) in (("ㄴ", "ㄹ"), ("ㄹ", "ㄴ")):
        return "表にない:流音化"
    if src == "語幹" and cb in "ㄱㄷㅅㅈ" and ja in ("ㄴ", "ㄵ", "ㅁ", "ㄻ", "ㄼ", "ㄾ"):
        return "パッチムの後の濃音化"                               # 標準発音法 第24・25項(有声音化より優先)
    if ja in ("ㄴ", "ㄹ", "ㅁ", "ㅇ", "ㄵ", "ㄻ", "ㄽ") and cb in "ㄱㄷㅂㅈ":
        return "有声音化"
    return None


CONJ_KEEP = "形が変わらない"


def conj_type(p, e):
    """conj_freq.py と同じ種類名(縮約で境目がない形の区分)"""
    tg, stem = bt(p.tag), p.form
    if tg == "EP":
        return "縮約:先語末語尾との融合"
    last = sy(stem[-1]) if stem else None
    if tg in ("VCP", "VCN"):
        return "縮約:이다・아니다"
    if stem == "하":
        return "縮約:하다→해"
    if p.tag.endswith("-I"):
        if stem.endswith("르"):
            return "縮約:르不規則"
        return {"ㅂ": "縮約:ㅂ不規則", "ㄷ": "縮約:ㄷ不規則", "ㅅ": "縮約:ㅅ不規則", "ㅎ": "縮約:ㅎ不規則"}.get(last[1] if last else "", "縮約:その他の不規則")
    o = ord(stem[-1]) - 0xAC00 if stem else -1
    v = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"[o % 588 // 28] if 0 <= o < 11172 else ""
    if v == "ㅡ":
        return "縮約:르不規則" if stem.endswith("르") and stem not in ("따르", "치르", "들르", "잇따르") else "縮約:으の脱落"
    return {"ㅏ": "縮約:母音縮約(ㅏ・ㅓ:同じ母音が重なる)", "ㅓ": "縮約:母音縮約(ㅏ・ㅓ:同じ母音が重なる)",
            "ㅗ": "縮約:母音縮約(ㅗ・ㅜ→ㅘ・ㅝ)", "ㅜ": "縮約:母音縮約(ㅗ・ㅜ→ㅘ・ㅝ)", "ㅣ": "縮約:母音縮約(ㅣ→ㅕ)",
            "ㅚ": "縮約:母音縮約(ㅚ→ㅙ)"}.get(v, "縮約:母音縮約(ㅐ・ㅔなど)")


def phon(t, core, end, j, e_end, ph_i):
    """判定範囲 = 語幹の最後の音節 + 語尾全体(言い回しなら言い回しの頭から)。
    戻り値 dict: 縮約(区分名 or ""), 組(音節の組と種類), 種類(重複を除いた集合), 注記"""
    first = core[ph_i] if ph_i is not None else core[end]
    prev = None
    for x in reversed(core[:core.index(first)]):
        if x.len > 0:
            prev = x
            break
    note = ""
    fused = ""
    if first.form[:1] in CJAMO or first.form[:1] in "ㄴㄹㅂㅁㅆ":
        a_idx = first.start                                           # 語尾の頭がパッチム(갈게・갑니다)
    elif prev is not None and first.start < prev.start + prev.len:
        a_idx = first.start                                           # 語幹と溶け合っている(가・해・봐)
        fused = conj_type(prev, first)
    else:
        a_idx = first.start - 1
        while a_idx >= 0 and t[a_idx] == " ":
            a_idx -= 1
            note = "境目に空白"
        if a_idx < 0 or not sy(t[a_idx]):
            a_idx = first.start
            note = "境目なし(直前が音節でない)"
    syl = [(k, t[k]) for k in range(a_idx, e_end) if sy(t[k])]

    def src_of(k):
        for x in core:
            if x.start == k and (x.form[:1] in CJAMO or x.form[:1] in "ㄴㄹㅂㅁㅆ"):
                return "語尾ㄹ" if x.form[0] in "ᆯㄹ" else "語尾"
        for x in core:
            if x.start == k and x.len and nf(x.form)[:1] == "을" and bt(x.tag).startswith("E"):
                return "語尾ㄹ"
        cov = [x for x in core if x.len and x.start <= k < x.start + x.len]
        if cov and bt(cov[-1].tag) in ("VV", "VA", "VX"):
            return "語幹"
        return "語尾"

    def same_stem(k1, k2):
        return any(x.len and bt(x.tag) in ("VV", "VA", "VX") and x.start <= k1 and k2 < x.start + x.len for x in core)

    pairs = []
    for (k1, s1), (k2, s2) in zip(syl, syl[1:]):
        src = src_of(k1)
        if src == "語幹" and same_stem(k1, k2):
            src = "語幹内"                                         # 안기다 の 안|기 は第24項の対象外
        ty = pair_type(s1, s2, src)
        if ty:
            pairs.append(f"{s1}|{s2}:{ty}")
    types = sorted({p.split(":", 1)[1] for p in pairs})
    return {"fused": fused, "pairs": pairs, "types": types, "note": note}


PHON_TESTS = ["했어", "가", "해", "봐요", "그랬거든요", "갈게요", "먹을게요", "할걸", "할지", "먹습니다", "갑니다", "했습니다", "좋아요", "좋다",
              "좋네요", "그렇지", "괜찮아", "싫어", "안다", "간다", "참지", "젊지", "감다", "안기다", "알지", "핥지", "좋습니다", "생각해", "못해", "감아", "할 거야", "먹을 수 있어", "있잖아", "했네", "했지", "했다",
              "예뻐요", "학생이야", "저예요", "맞아", "읽어", "앉아", "넓다", "많네", "몰라", "추워", "들어", "나도요", "학생이요", "먹었어요", "됐어", "하셨어요", "보고 싶어"]


def phon_test(a):
    import kiwipiepy
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    print(f"# STEP5 判定の試し(kiwipiepy {kiwipiepy.__version__})\n\n| 例 | 語尾 | 縮約 | 音節の組と種類 | 注記 |\n|---|---|---|---|---|")
    for s in PHON_TESTS:
        d = extract(kiwi, [("", 0, s)], "t")[0]
        if d["kind"] != "語尾":
            print(f"| {s} | ({d['kind']}) | | | |")
            continue
        for lab, ph in (("", d["phon"]), ("言い回し " + d["phrase"], d.get("phon_ph"))):
            if ph is None:
                continue
            print(f"| {s} | {lab or d['item2']} | {ph['fused'] or '—'} | {'、'.join(ph['pairs']) or '変化なし'} | {ph['note']} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["survey", "telop", "draft", "phon_test"])
    ap.add_argument("--gose", default="seventeen_analysis/output/ani_textbook_pipeline/corpus.csv")
    ap.add_argument("--bts", default="bts_pipeline/bts_textbook_pipeline/bts_corpus.csv")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="gobi_analysis")
    a = ap.parse_args()
    {"survey": survey, "telop": telop_check, "draft": draft, "phon_test": phon_test}[a.stage](a)


if __name__ == "__main__":
    main()
