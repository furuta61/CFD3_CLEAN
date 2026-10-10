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


# ------------------------------------------------------------------ STEP1・2
PRED = {"VV", "VA", "VX", "VCP", "VCN", "XSV", "XSA"}
NOUNISH = {"NNG", "NNP", "NNB", "NR", "NP", "MAG", "MAJ"}
NO_STRIP = {"세요", "에요", "예요"}
# 으・은・을・읍・는 の有無(まとめ方 final と同じ)
F_EF = {"을까": "ㄹ까", "을게": "ㄹ게", "을래": "ㄹ래", "으세요": "세요", "으세": "세요", "은가": "ㄴ가", "는가": "ㄴ가",
        "은데": "ㄴ데", "는데": "ㄴ데", "는다": "ㄴ다", "으냐": "냐", "느냐": "냐", "ㅂ니다": "습니다", "ㅂ니까": "습니까",
        "읍시다": "ㅂ시다", "으니": "니", "으라": "라"}
F_EC = {"으면": "면", "은데": "ㄴ데", "는데": "ㄴ데", "으니까": "니까", "으러": "러", "으려고": "려고", "을게": "ㄹ게"}
ETM_N = {"을": "ㄹ", "ㄹ": "ㄹ", "은": "ㄴ", "ㄴ": "ㄴ", "는": "는", "던": "던"}


def norm_ending(tag, form):
    """(項目, 適用した規則のリスト)"""
    rules = []
    f = nf(form)
    if f == "죠":
        f = "지"
        rules.append("縮約(죠=지요)")
    elif f.endswith("요") and len(f) > 1 and f not in NO_STRIP:
        f = f[:-1]
        rules.append("요の有無")
    if f[0] in "아여" and len(f) >= 1:
        g = "어" + f[1:]
        if g != f:
            f = g
            rules.append("아/어/여")
    m = (F_EF if tag == "EF" else F_EC).get(f)
    if m:
        f = m
        rules.append("으・는などの有無")
    return f, rules


def phrase_of(core, j, end):
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
        if pt == "VX" or (ec == "어야" and pf == "되") or (ec == "어도" and pf == "되") or (ec == "게" and pf == "되"):
            return f"-{ec} {pf}다", j - 1, False
    return None, None, False


def extract(kiwi, items, corpus):
    """1行ずつ:(行情報, 結果 dict)"""
    rows = []
    for (ep, row, t), toks in zip(items, kiwi.tokenize([x[2] for x in items])):
        core = [x for x in toks if not x.tag.startswith("S")]
        d = {"corpus": corpus, "ep": ep, "row": row, "text": t, "an": " ".join(f"{x.form}/{x.tag}" for x in core)}
        if not core:
            d["kind"] = "除外:形態素なし"
            rows.append(d)
            continue
        end = len(core) - 1
        yo_tail = None
        if end > 0 and core[end].form == "요" and core[end].tag in ("JX", "MM"):
            yo_tail = core[end]
            end -= 1
        e = core[end]
        tg = bt(e.tag)
        if tg == "EF" and nf(e.form) == "요" and end >= 1 and bt(core[end - 1].tag) in ("EF", "EC"):
            yo_tail = e          # 単独の 요(EF) は直前の語尾の 요 とみなす(まとめ方 final と同じ)
            end -= 1
            e = core[end]
            tg = bt(e.tag)
        if tg not in ("EF", "EC"):
            d["kind"] = f"除外:その他の語尾({tg})" if tg.startswith("E") else f"除外:語尾なし({tg})"
            rows.append(d)
            continue
        if tg == "EF" and nf(e.form) == "요":
            d["kind"] = "除外:単独の요(EF)"
            rows.append(d)
            continue
        j = end - 1
        eps = []
        while j >= 0 and bt(core[j].tag) == "EP":
            eps.insert(0, nf(core[j].form))
            j -= 1
        surf = nf(e.form) + ("+요" if yo_tail is not None else "")
        item, rules = norm_ending(tg, e.form)
        if yo_tail is not None and "요の有無" not in rules:
            rules.insert(0, "요の有無")
        if item in ("에요", "예요"):
            if j >= 0 and bt(core[j].tag) in ("VCP", "VCN"):
                item = "야"
                rules.append("에요/예요→야(指定詞の後)")
            else:
                rules.append("要確認:에요/예요が指定詞の後でない")
        d.update(kind="語尾", tag=tg, surf=surf, item=item, rules="・".join(rules), ep_list="+".join(eps),
                 pred="", pred_ok=True, phrase="", phrase_rec=False)
        if j >= 0 and bt(core[j].tag) in PRED:
            p = core[j]
            d["pred"] = f"{nf(p.form)}/{p.tag}"
            # 句の中の EP(할 거였어 の 었 は 이 の後)も上で拾える。거 같았어 も同じ
            ph, st_i, rec = phrase_of(core, j, end)
            if ph:
                d["phrase"], d["phrase_rec"] = ph, rec
                d["ph_start"] = core[st_i].start
        else:
            d["pred_ok"] = False
            d["pred"] = f"{core[j].form}/{core[j].tag}" if j >= 0 else "(なし)"
        last = yo_tail if yo_tail is not None else e
        d["e_start"], d["e_end"] = e.start, last.start + last.len
        d["spell"] = t[e.start:last.start + last.len]
        rows.append(d)
    return rows


def draft(a):
    import kiwipiepy
    from kiwipiepy import Kiwi
    o = outdir(os.path.join(a.out, "step1_2"))
    kiwi = Kiwi()
    _, gst, _, g = load_gose(a.gose)
    _, bst, _, b = load_bts(a.bts)
    data = {"GOSE": extract(kiwi, g, "GOSE"), "run": extract(kiwi, b.get("run", []), "run"),
            "bomb": extract(kiwi, b.get("bomb", []), "bomb")}
    R = [f"# STEP1・2 抽出と統合表の案(集計前・要確認)\n\nkiwipiepy {kiwipiepy.__version__} / 前処理V2 / 1行=1発話、行末で判定\n"]

    # 抽出結果(STEP3 以降はこれを読む)
    head = ["corpus", "話", "行番号", "行", "区分", "タグ", "表層形", "項目(語尾)", "規則", "先語末語尾", "直前の用言",
            "直前が用言", "言い回し候補", "言い回し推奨", "語尾開始位置", "語尾終了位置", "言い回し開始位置", "綴り", "解析"]
    for n, rs in data.items():
        wcsv(os.path.join(o, f"extract_{n}.csv"), head,
             [[d["corpus"], d["ep"], d["row"], d["text"], d["kind"], d.get("tag", ""), d.get("surf", ""), d.get("item", ""),
               d.get("rules", ""), d.get("ep_list", ""), d.get("pred", ""), int(d.get("pred_ok", False)) if d["kind"] == "語尾" else "",
               d.get("phrase", ""), int(d.get("phrase_rec", False)) if d.get("phrase") else "", d.get("e_start", ""),
               d.get("e_end", ""), d.get("ph_start", ""), d.get("spell", ""), d["an"]] for d in rs])

    # 行の内訳(語尾が特定できた行 / 除外)
    R.append("## STEP1 行末の内訳(分母:発話行)\n\n| 区分 | GOSE | Run BTS | BOMB |\n|---|---:|---:|---:|")
    kc = {n: Counter(d["kind"] for d in rs) for n, rs in data.items()}
    kinds = sorted(set().union(*kc.values()), key=lambda k: (k != "語尾", -kc["GOSE"][k]))
    for k in kinds:
        R.append(f"| {k} | " + " | ".join(f"{kc[n][k]} ({kc[n][k] / max(len(data[n]), 1) * 100:.1f}%)" for n in data) + " |")
    R.append("| 発話行(分母) | " + " | ".join(str(len(data[n])) for n in data) + " |")
    npo = {n: sum(1 for d in rs if d["kind"] == "語尾" and not d["pred_ok"]) for n, rs in data.items()}
    R.append("\n- 語尾の直前が用言・先語末語尾でない行(kiwiの誤りの疑い。今は「語尾」に含めたまま。要確認): "
             + " / ".join(f"{n} {npo[n]}" for n in data))

    # 言い回し候補
    pc = defaultdict(lambda: {"GOSE": 0, "run": 0, "bomb": 0, "rec": False, "ex": "", "ends": Counter()})
    for n, rs in data.items():
        for d in rs:
            if d.get("phrase"):
                x = pc[d["phrase"]]
                x[n] += 1
                x["rec"] = d["phrase_rec"]
                x["ends"][d["item"]] += 1
                if not x["ex"]:
                    x["ex"] = d["text"]
    ph_rows = sorted(pc.items(), key=lambda kv: -(kv[1]["GOSE"] + kv[1]["run"] + kv[1]["bomb"]))
    wcsv(os.path.join(o, "phrase_candidates.csv"),
         ["言い回し", "GOSE", "Run BTS", "BOMB", "推奨(○=1項目として拾う案)", "後ろの語尾の内訳(上位5)", "例", "採用(○/×。空欄=未確認)"],
         [[k, v["GOSE"], v["run"], v["bomb"], "○" if v["rec"] else "要確認",
           "、".join(f"{e}{c}" for e, c in v["ends"].most_common(5)), v["ex"], ""] for k, v in ph_rows if
          v["GOSE"] + v["run"] + v["bomb"] >= 5])
    R.append(f"\n## STEP1 語をまたぐ言い回しの候補 → phrase_candidates.csv(合計5件以上 {sum(1 for _, v in ph_rows if v['GOSE'] + v['run'] + v['bomb'] >= 5)}種類)\n")
    R.append("| 言い回し | GOSE | Run BTS | BOMB | 推奨 |\n|---|---:|---:|---:|---|")
    for k, v in ph_rows[:25]:
        R.append(f"| {k} | {v['GOSE']} | {v['run']} | {v['bomb']} | {'○' if v['rec'] else '要確認'} |")

    # 統合表(言い回しは別の表。ここは語尾の表層形→項目)
    mt = defaultdict(lambda: {"GOSE": 0, "run": 0, "bomb": 0, "npo": 0, "spell": Counter(), "ex": "", "rules": ""})
    item_tags = defaultdict(Counter)
    for n, rs in data.items():
        for d in rs:
            if d["kind"] != "語尾":
                continue
            x = mt[(d["tag"], d["surf"], d["item"])]
            x[n] += 1
            x["npo"] += not d["pred_ok"]
            x["spell"][d["spell"]] += 1
            x["rules"] = d["rules"]
            if not x["ex"]:
                x["ex"] = d["text"]
            item_tags[d["item"]][d["tag"]] += 1
    rows = []
    flagged = 0
    for (tg, sf, it), v in sorted(mt.items(), key=lambda kv: -(kv[1]["GOSE"] + kv[1]["run"] + kv[1]["bomb"])):
        tot = v["GOSE"] + v["run"] + v["bomb"]
        why = []
        if "要確認" in v["rules"]:
            why.append(v["rules"].split("要確認:")[1])
        if it in NO_STRIP or it.endswith("요"):
            why.append("요を外さない形")
        tg_c = item_tags[it]
        if len(tg_c) > 1 and min(tg_c.values()) / sum(tg_c.values()) >= 0.1:
            why.append(f"同じ項目にEFとECが混在(EF{tg_c['EF']}/EC{tg_c['EC']})")
        if v["npo"] / tot >= 0.2:
            why.append(f"直前が用言でない {v['npo']}件")
        if it in ("잖아", "대", "래", "냬", "재"):
            why.append("縮約形(-지 않아/-다고 해 など)と別項目のままにしている")
        flagged += bool(why)
        rows.append([sf, tg, it, v["rules"], v["GOSE"], v["run"] + v["bomb"], v["run"], v["bomb"],
                     "、".join(f"{s}{c}" for s, c in v["spell"].most_common(3)), v["ex"], "要確認:" + "/".join(why) if why else ""])
    wcsv(os.path.join(o, "merge_table.csv"),
         ["表層形(kiwi)", "タグ", "項目", "適用した規則", "GOSE", "BTS計", "Run BTS", "BOMB", "字幕上の綴り(上位3)", "例", "要確認"], rows)
    n_item = len({r[2] for r in rows})
    R.append(f"\n## STEP2 統合表の案 → merge_table.csv\n\n- 表層形 {len(rows)} 行 → 項目 {n_item} 個 / 要確認の印 {flagged} 行")
    R.append("- 規則: 요の有無・죠=지요・아/어/여(kiwiは 아→어 を既に統一)・으/는などの有無(まとめ方 final と同じ表)・에요/예요→야(指定詞の後)")
    R.append("- 해 は kiwi では 하+어 に分かれるので、項目「어」に入る(先語末語尾・用言の列で区別できる)")
    R.append("\n### 項目の上位30(言い回しを当てる前。GOSE件数順)\n\n| 項目 | GOSE | BTS計 | 表層形の数 |\n|---|---:|---:|---:|")
    agg = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        agg[r[2]][0] += r[4]
        agg[r[2]][1] += r[5]
        agg[r[2]][2] += 1
    for it, (gc, bc, ns) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:30]:
        R.append(f"| {it} | {gc} | {bc} | {ns} |")
    R.append("\n### 要確認の行(合計件数の上位20)\n\n| 表層形 | タグ | 項目 | GOSE | BTS計 | 理由 |\n|---|---|---|---:|---:|---|")
    for r in [r for r in rows if r[-1]][:20]:
        R.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[4]} | {r[5]} | {r[-1][4:]} |")
    R.append("\n確認してほしいもの: phrase_candidates.csv の採用欄、merge_table.csv の項目と要確認の行。確認が済むまで STEP3 以降は実行しない。")
    txt = "\n".join(R)
    open(os.path.join(o, "step1_2_report.md"), "w", encoding="utf-8").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["survey", "draft"])
    ap.add_argument("--gose", required=True)
    ap.add_argument("--bts", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="gobi_analysis")
    a = ap.parse_args()
    {"survey": survey, "draft": draft}[a.stage](a)


if __name__ == "__main__":
    main()
