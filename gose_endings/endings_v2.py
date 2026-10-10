#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""語尾のまとめ方 v2 の再集計と、記録の食い違いの確認(第三弾 企画判断用)

v1 = part1_4_endings.py / bts_compare.py と同じまとめ方。
v2 = v1 に次を追加:
  ・直前が VCP/VCN の 에요/예요 → 야(이에요↔이야、아니에요↔아니야)。それ以外の 에요/예요 は別項目のまま
  ・行末 EF の 요:直前が語尾(EF/EC)なら 요 を外して直前の語尾を行末とする。
    直前が名詞・代名詞・副詞・感嘆詞なら「語尾なし」。どちらでもなければ「未区分」として件数と用例を出す
既存の出力は上書きしない(出力先 --out、既定 out_endings_v2)。kpop_analyze.py は実行しない(PART6 は読むだけ)。

使い方(コーパス分析フォルダで):
  python3 endings_v2.py --gose seventeen_analysis/output/ani_textbook_pipeline/corpus.csv \
     --bts bts_pipeline/bts_textbook_pipeline/bts_corpus.csv --utt-dir seventeen_analysis \
     --kpop ~/Downloads/書籍/kpop_analyze.py --out out_endings_v2
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

import kiwipiepy
from kiwipiepy import Kiwi

BR = re.compile(r"\[[^\]]*\]")
PA = re.compile(r"\([^)]*\)")
HANGUL = re.compile(r"[가-힣]")
JAMO = str.maketrans({"ᆫ": "ㄴ", "ᆯ": "ㄹ", "ᆸ": "ㅂ", "ᆷ": "ㅁ", "ᆻ": "ㅆ"})
NO_STRIP = {"세요", "에요", "예요"}
MERGE = {"ㅂ니다": "습니다", "을까": "ㄹ까"}
GOSE14_V1 = ["어", "야", "지", "습니다", "다", "네", "잖아", "세요", "자", "ㄹ게", "ㄹ까", "ㄴ다", "ㄴ가", "거든"]
NOUNISH = ("NNG", "NNP", "NNB", "NR", "NP", "MAG", "MAJ", "IC", "XR")
CATS = ("EF", "EC", "その他", "語尾なし")
P0 = {"GOSE": (102006, 50783, 44020), "run": (120127, 57388, 47396), "bomb": (35907, 16684, 14083)}
SEED = 42


def clean(t):
    return re.sub(r"\s+", " ", PA.sub(" ", BR.sub(" ", t))).strip()


def nf(f):
    return f.translate(JAMO)


def strip_yo(form):
    form = nf(form)
    if form == "죠":
        return "지", True
    if form in NO_STRIP:
        return form, True
    if form.endswith("요") and len(form) > 1:
        return form[:-1], True
    return form, False


def bt(tag):
    return tag.split("-")[0]


def an(core):
    return " ".join(f"{t.form}/{t.tag}" for t in core)


def pct(a, b, d=1):
    return f"{a / b * 100:.{d}f}%" if b else "—"


class Res:
    def __init__(self):
        self.lines = 0
        self.morph = 0
        self.morph_all = 0
        self.cat = {"v1": Counter(), "v2": Counter()}
        self.ef = {"v1": Counter(), "v2": Counter()}
        self.yo = {"v1": Counter(), "v2": Counter()}
        self.eps = {"v1": defaultdict(set), "v2": defaultdict(set)}
        self.ec2 = Counter()
        self.eyo_prev = Counter()
        self.eyo_ex = defaultdict(list)
        self.yo_prev = Counter()
        self.yo_ex = []
        self.yo_forms = Counter()
        self.unclassified = []
        self.ge = Counter()          # 거든 の数え方別


def analyze(kiwi, items):
    R = Res()
    data = [(ep, clean(t)) for ep, t in items]
    data = [(ep, t) for ep, t in data if HANGUL.search(t)]
    R.lines = len(data)
    for (ep, t), toks in zip(data, kiwi.tokenize([x[1] for x in data])):
        R.morph_all += len(toks)
        core = [x for x in toks if not x.tag.startswith("S")]
        R.morph += len(core)
        for x in core:
            if x.tag == "EF" and nf(x.form) == "거든":
                R.ge["거든_any"] += 1
            if x.tag == "EF" and nf(x.form) == "거든요":
                R.ge["거든요_any"] += 1
            if x.tag == "EC" and nf(x.form) in ("거든", "거든요"):
                R.ge["거든EC_any"] += 1
        if not core:
            continue
        end = len(core) - 1
        yo_tail = False
        if end > 0 and core[end].form == "요" and core[end].tag in ("JX", "MM"):
            yo_tail = True
            end -= 1
        last = core[end]
        tag = bt(last.tag)
        # ---- v1
        if tag.startswith("E"):
            k1 = tag if tag in ("EF", "EC") else "その他"
            R.cat["v1"][k1] += 1
            if k1 == "EF":
                raw = nf(last.form)
                if raw.endswith("요") or raw == "죠":
                    R.yo_forms[raw] += 1
                f, yo = strip_yo(last.form)
                f = MERGE.get(f, f)
                R.ef["v1"][f] += 1
                R.yo["v1"][f] += int(yo or yo_tail)
                R.eps["v1"][f].add(ep)
        else:
            k1 = "語尾なし"
            R.cat["v1"][k1] += 1
        # ---- v2
        k2, f2, yo2 = k1, None, False
        if k1 == "EF":
            f2, yo2 = strip_yo(last.form)
            f2 = MERGE.get(f2, f2)
            yo2 = yo2 or yo_tail
            prev = core[end - 1] if end >= 1 else None
            if f2 in ("에요", "예요"):
                pt = bt(prev.tag) if prev else "(なし)"
                key = pt if pt in ("VCP", "VCN") else "その他"
                R.eyo_prev[(f2, key, pt)] += 1
                if len(R.eyo_ex[(f2, key)]) < 400:
                    R.eyo_ex[(f2, key)].append((ep, t, an(core)))
                if key in ("VCP", "VCN"):
                    f2 = "야"
            elif nf(last.form) == "요":
                pt = bt(prev.tag) if prev else "(なし)"
                R.yo_prev[pt] += 1
                R.yo_ex.append((ep, t, an(core), pt))
                if prev is not None and pt in ("EF", "EC"):
                    k2 = pt
                    f2, _ = strip_yo(prev.form)
                    f2 = MERGE.get(f2, f2)
                    yo2 = True
                elif prev is not None and pt in NOUNISH:
                    k2, f2 = "語尾なし", None
                else:
                    k2, f2 = "未区分", None
                    R.unclassified.append((ep, t, an(core), pt))
        R.cat["v2"][k2] += 1
        if k2 == "EF":
            R.ef["v2"][f2] += 1
            R.yo["v2"][f2] += int(yo2)
            R.eps["v2"][f2].add(ep)
        elif k2 == "EC":
            f3 = f2 if f2 else strip_yo(last.form)[0]
            R.ec2[f3] += 1
        if k2 == "EF" and f2 == "거든":
            R.ge["거든_final"] += 1
    return R


def ranks(c):
    return {f: i for i, (f, _) in enumerate(c.most_common(), 1)}


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** .5
    sy = sum((y - my) ** 2 for y in ys) ** .5
    return sxy / (sx * sy) if sx and sy else None


def write_csv(path, head, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gose", required=True)
    ap.add_argument("--bts", required=True)
    ap.add_argument("--utt-dir")
    ap.add_argument("--kpop", default=os.path.expanduser("~/Downloads/書籍/kpop_analyze.py"))
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="out_endings_v2")
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit(f"{a.out} に既にファイルがあります。上書きしないため停止します。")
    os.makedirs(a.out, exist_ok=True)
    O = [f"# まとめ方v2 再集計\n\nkiwipiepy {kiwipiepy.__version__} / 前処理 V2\n"]
    kiwi = Kiwi()
    rnd = random.Random(SEED)

    with open(a.gose, encoding="utf-8-sig", newline="") as f:
        g_raw = list(csv.DictReader(f))
    with open(a.bts, encoding="utf-8-sig", newline="") as f:
        b_raw = list(csv.DictReader(f))
    items = {
        "GOSE": [(r.get("episode", ""), r.get("text") or "") for r in g_raw],
        "run": [(r.get("clip_id", ""), r.get("text") or "") for r in b_raw if r.get("source") == "run"],
        "bomb": [(r.get("clip_id", ""), r.get("text") or "") for r in b_raw if r.get("source") == "bomb"],
    }
    items["run+bomb"] = items["run"] + items["bomb"]
    res = {k: analyze(kiwi, v) for k, v in items.items()}

    # ================= PART0
    O.append("## PART0 再現確認(v1)\n\n| コーパス | 項目 | 期待値 | 今回 | 一致 |\n|---|---|---:|---:|---|")
    ok = True
    for k, (n, ef, s14) in P0.items():
        X = res[k]
        got = (X.lines, X.cat["v1"]["EF"], sum(X.ef["v1"][f] for f in GOSE14_V1))
        for lab, e, g in zip(("全行", "EF", "GOSEの14種類(v1)"), (n, ef, s14), got):
            O.append(f"| {k} | {lab} | {e} | {g} | {'○' if e == g else '×'} |")
            ok &= e == g
    if not ok:
        O.append("\n**不一致のため PART1 以降は実行していない。**")
        return finish(a, O)
    O.append("\n一致。")

    # ================= PART1
    O.append("\n## PART1 まとめ方の問題の中身\n")
    for k in ("GOSE", "run", "bomb"):
        X = res[k]
        O.append(f"### {k}\n\n**1. 行末EFが 에요/예요 の行の直前タグ**\n\n| 形 | 区分 | 直前タグ | 件数 |\n|---|---|---|---:|")
        for (f, key, pt), v in sorted(X.eyo_prev.items(), key=lambda kv: -kv[1]):
            O.append(f"| {f} | {key} | {pt} | {v} |")
        ex = []
        for (f, key), lst in X.eyo_ex.items():
            for e in rnd.sample(lst, min(10, len(lst))):
                ex.append([f, key, *e])
        write_csv(os.path.join(a.out, f"part1_eyo_examples_{k}.csv"), ["形", "区分", "話", "行", "解析"], ex)
        O.append(f"\n用例: part1_eyo_examples_{k}.csv(形×区分ごとに最大10行)\n\n**2. 行末EFが 요 の行の直前タグ**(合計 {sum(X.yo_prev.values())})\n\n| 直前タグ | 件数 |\n|---|---:|")
        for pt, v in X.yo_prev.most_common():
            O.append(f"| {pt} | {v} |")
        smp = rnd.sample(X.yo_ex, min(20, len(X.yo_ex)))
        write_csv(os.path.join(a.out, f"part1_yo_examples_{k}.csv"), ["話", "行", "解析", "直前タグ"], smp)
        O.append(f"\n用例: part1_yo_examples_{k}.csv({len(smp)}行)\n\n**3. 요 が語尾と一体になった行末EF**\n\n| 形 | 件数 | v1 での数え方 |\n|---|---:|---|")
        for f, v in X.yo_forms.most_common(30):
            if f == "죠":
                how = "지 に統合(요付き)"
            elif f in NO_STRIP:
                how = "そのまま別項目(요を外さない)"
            elif f == "요":
                how = "요 のまま別項目"
            else:
                how = f"요 を外して {MERGE.get(f[:-1], f[:-1])} に統合(요付き)"
            O.append(f"| {f} | {v} | {how} |")
        O.append("")

    # ================= PART2
    O.append("## PART2 まとめ方v2 の再集計\n")
    g2 = [f for f, _ in res["GOSE"].ef["v2"].most_common(14)]
    O.append("GOSE の上位14(v2): " + "、".join(g2) + "\n")
    summary = {}
    for k in ("GOSE", "run", "bomb", "run+bomb"):
        X = res[k]
        n = X.lines
        O.append(f"### {k}(全行 {n})\n\n| 区分 | v1 行数 | v1 割合 | v2 行数 | v2 割合 |\n|---|---:|---:|---:|---:|")
        for c in CATS + ("未区分",):
            O.append(f"| {c} | {X.cat['v1'][c]} | {pct(X.cat['v1'][c], n)} | {X.cat['v2'][c]} | {pct(X.cat['v2'][c], n)} |")
        if X.unclassified:
            write_csv(os.path.join(a.out, f"part2_unclassified_{k}.csv"), ["話", "行", "解析", "直前タグ"],
                      rnd.sample(X.unclassified, min(30, len(X.unclassified))))
            O.append(f"\n未区分の用例: part2_unclassified_{k}.csv")
        ef = X.cat["v2"]["EF"]
        rows = []
        cum = 0
        O.append(f"\nEF上位30(v2、分母:EF行 {ef})\n\n| 順位 | 形 | 件数 | EF比 | 累積 | 요付き | 話数 |\n|---:|---|---:|---:|---:|---:|---:|")
        for i, (f, c) in enumerate(X.ef["v2"].most_common(30), 1):
            cum += c
            O.append(f"| {i} | {f} | {c} | {pct(c, ef)} | {pct(cum, ef)} | {X.yo['v2'][f]} | {len(X.eps['v2'][f])} |")
            rows.append([i, f, c, c / ef, cum / ef, X.yo["v2"][f], len(X.eps["v2"][f])])
        write_csv(os.path.join(a.out, f"ef_top30_v2_{k}.csv"), ["順位", "形", "件数", "EF比", "累積", "요付き", "話数"], rows)
        for ver, g14 in (("v1", GOSE14_V1), ("v2", g2)):
            efv = X.cat[ver]["EF"]
            s = sum(X.ef[ver][f] for f in g14)
            own = sum(c for _, c in X.ef[ver].most_common(14))
            summary[(k, ver)] = (s, own, efv)
        O.append("\n| 占有率 | v1 | v2 |\n|---|---:|---:|")
        for lab, idx in (("GOSEの14種類", 0), ("自身の上位14種類", 1)):
            v1, v2 = summary[(k, "v1")], summary[(k, "v2")]
            O.append(f"| {lab} | {v1[idx]}/{v1[2]} = {pct(v1[idx], v1[2])} | {v2[idx]}/{v2[2]} = {pct(v2[idx], v2[2])} |")
        O.append("")
    O.append("### 重なりと判定\n\n| 比較 | 版 | 上位14入り | 上位20入り | スピアマン(和集合) | 占有率 | 基準(GOSE値−5) | 判定 |\n|---|---|---:|---:|---:|---:|---:|---|")
    G = res["GOSE"]
    for k in ("run", "bomb", "run+bomb"):
        X = res[k]
        for ver, g14 in (("v1", GOSE14_V1), ("v2", g2)):
            b14 = [f for f, _ in X.ef[ver].most_common(14)]
            b20 = [f for f, _ in X.ef[ver].most_common(20)]
            i14, i20 = sum(f in b14 for f in g14), sum(f in b20 for f in g14)
            gr, br = ranks(G.ef[ver]), ranks(X.ef[ver])
            union = list(dict.fromkeys(g14 + b14))
            rho = spearman([gr.get(f, len(gr) + 1) for f in union], [br.get(f, len(br) + 1) for f in union])
            s, _, efv = summary[(k, ver)]
            gs, _, gef = summary[("GOSE", ver)]
            thr = gs / gef * 100 - 5
            occ = s / efv * 100
            c1, c2 = occ >= thr, i20 >= 12
            v = "成立" if c1 and c2 else "部分的に成立" if c1 or c2 else "不成立"
            if k != "run":
                v += "(参考)"
            O.append(f"| GOSE×{k} | {ver} | {i14}/14 | {i20}/14 | {'—' if rho is None else f'{rho:.3f}'} | "
                     f"{occ:.1f}% | {thr:.1f}% | {v} |")
    # 7. 企画書の数値
    O.append("\n### 企画書v3 の数値の v1 → v2\n\n| 項目 | 分母 | v1 | v2 |\n|---|---|---:|---:|")
    gv1, gv2 = summary[("GOSE", "v1")], summary[("GOSE", "v2")]
    O.append(f"| 上位14の占有率(87%) | 行末がEFの行 | {gv1[0]}/{gv1[2]} = {pct(gv1[0], gv1[2])} | {gv2[0]}/{gv2[2]} = {pct(gv2[0], gv2[2])} |")
    O.append(f"| 上位14の全行比(43%) | 全行 {G.lines} | {pct(gv1[0], G.lines)} | {pct(gv2[0], G.lines)} |")
    O.append(f"| 어 の割合(35%) | 行末がEFの行 | {pct(G.ef['v1']['어'], gv1[2])} | {pct(G.ef['v2']['어'], gv2[2])} |")
    top5ec_v2 = sum(c for _, c in G.ec2.most_common(5))
    O.append(f"| 上位14+行末EC上位5+語尾なし(87%) | 全行 {G.lines} | 88632 = {pct(88632, G.lines)}(既存値) | "
             f"{gv2[0] + top5ec_v2 + G.cat['v2']['語尾なし']} = {pct(gv2[0] + top5ec_v2 + G.cat['v2']['語尾なし'], G.lines)} |")
    O.append(f"\nv2 の行末EC上位5: " + "、".join(f"{f} {c}" for f, c in G.ec2.most_common(5)))
    O.append("\n累積カバー率の列(行末がEFの行、上位14)\n\n| 順位 | v1 | v2 |\n|---:|---|---|")
    c1 = c2 = 0
    l1, l2 = G.ef["v1"].most_common(14), G.ef["v2"].most_common(14)
    for i in range(min(14, len(l1), len(l2))):
        c1 += l1[i][1]
        c2 += l2[i][1]
        O.append(f"| {i + 1} | {l1[i][0]} {pct(c1, gv1[2])} | {l2[i][0]} {pct(c2, gv2[2])} |")

    # ================= PART3
    O.append("\n## PART3 Run BTS の年の情報\n")
    cands = []
    for dp, dn, fn in os.walk(a.root):
        if "bts" not in dp.lower() and "bts" not in " ".join(fn).lower():
            continue
        for f in fn:
            if not f.endswith((".py", ".csv", ".json", ".md")):
                continue
            p = os.path.join(dp, f)
            try:
                if os.path.getsize(p) > 50_000_000:
                    continue
                s = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if "2015" in s and ("2021" in s or "year" in s or "通時" in s or "연도" in s):
                cands.append(p)
    run_clips = set(ep for ep, _ in items["run"])
    maps = []
    for p in cands:
        if p.endswith(".csv"):
            try:
                rs = list(csv.DictReader(open(p, encoding="utf-8-sig", errors="ignore")))
            except Exception:
                continue
            if not rs:
                continue
            cols = [c for c in rs[0].keys() if isinstance(c, str) and c]
            ycols = [c for c in cols if re.search(r"year|年|date|연도", c, re.I)]
            for ic in cols:
                vals = set(str(r.get(ic)) for r in rs)
                if len(run_clips & vals) >= .9 * len(run_clips):
                    for yc in ycols:
                        m = {}
                        for r in rs:
                            y = re.search(r"(20\d\d)", str(r.get(yc, "")))
                            if y:
                                m[str(r.get(ic))] = int(y.group(1))
                        if run_clips <= set(m):
                            maps.append((p, ic, yc, m))
        elif p.endswith(".json"):
            try:
                d = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            if isinstance(d, dict) and len(run_clips & set(d)) >= .9 * len(run_clips):
                m = {}
                for kk, v in d.items():
                    s = json.dumps(v, ensure_ascii=False)
                    y = re.search(r"(20\d\d)", s)
                    if y:
                        m[kk] = int(y.group(1))
                if run_clips <= set(m):
                    maps.append((p, "(キー)", "(値の中の4桁の年)", m))
    O.append(f"候補ファイル({len(cands)}件、最大40件表示):\n")
    for p in cands[:40]:
        O.append(f"- {p}")
    if len(maps) == 1:
        p, ic, yc, m = maps[0]
        O.append(f"\nクリップ→年の対応: {p}(クリップ列 {ic}、年の列 {yc})")
        for lab, cond in (("2015–2016", lambda y: y <= 2016), ("2017以降", lambda y: y >= 2017)):
            X = analyze(kiwi, [(ep, t) for ep, t in items["run"] if cond(m[ep])])
            for ver, g14 in (("v1", GOSE14_V1), ("v2", g2)):
                s = sum(X.ef[ver][f] for f in g14)
                O.append(f"- {lab} {ver}: GOSEの14種類 {s}/EF行 {X.cat[ver]['EF']} = {pct(s, X.cat[ver]['EF'])}(全行 {X.lines})")
    else:
        O.append(f"\n年別: 未実施(クリップ全体に年を割り当てられるファイル {len(maps)} 件。"
                 + ("複数あり正本を決められない: " + "; ".join(m[0] for m in maps) if maps else "見つからない") + ")")

    # ================= PART4
    O.append("\n## PART4 形態素数の分母\n")
    utt = []
    if a.utt_dir:
        for f in sorted(glob.glob(os.path.join(a.utt_dir, "*_utterances.json"))):
            d = json.load(open(f, encoding="utf-8"))
            segs = d if isinstance(d, list) else d.get("segments", d)
            for s in segs:
                tx = str(s.get("text", "")) if isinstance(s, dict) else str(s)
                utt.append((os.path.basename(f), tx.strip()))

    def count(texts, v2, sym):
        ts = [clean(x) if v2 else re.sub(r"\s+", " ", x).strip() for x in texts]
        ts = [x for x in ts if HANGUL.search(x)]
        n = 0
        for toks in kiwi.tokenize(ts):
            n += len(toks) if sym else sum(1 for x in toks if not x.tag.startswith("S"))
        return n, len(ts)
    O.append("| コーパス | 入力 | 前処理 | 記号 | 形態素数 | 行数 |\n|---|---|---|---|---:|---:|")
    conds = [("GOSE", "corpus.csv", [t for _, t in items["GOSE"]]),
             ("Run BTS", "bts_corpus.csv(run)", [t for _, t in items["run"]]),
             ("BOMB", "bts_corpus.csv(bomb)", [t for _, t in items["bomb"]])]
    if utt:
        conds.insert(1, ("GOSE", "utterances.json", [t for _, t in utt]))
    for name, src, texts in conds:
        for v2 in (True, False):
            for sym in (False, True):
                n, l = count(texts, v2, sym)
                O.append(f"| {name} | {src} | {'V2' if v2 else 'V0(空白のみ)'} | {'含む' if sym else '除く'} | {n} | {l} |")
    O.append("\n- scene_extraction.py(Google ドライブの版)の BTS 入力は bts_s1s2_whisper と bts_s3_whisper の .srt 144本のみ(Run BTS のみ、BOMB を含まない)。"
             "トークン数は kpop_analyze の kiwi で、記号を含めて数えている(len(kiwi.tokenize(utt)))。GOSE 入力は *_utterances.json、clean_text 適用、テロップ除外なし")
    O.append("- kpop_analyze の clean_text / kiwi を使った旧方式の完全再現: 未実施(PART6 の指示により kpop_analyze.py を実行していない)")
    hits = []
    for dp, dn, fn in os.walk(a.root):
        for f in fn:
            if f.endswith((".md", ".txt", ".log", ".csv")):
                p = os.path.join(dp, f)
                try:
                    if os.path.getsize(p) > 5_000_000:
                        continue
                    for i, line in enumerate(open(p, encoding="utf-8", errors="ignore")):
                        if re.search(r"(82\.1万|821,\d{3}|72\.7万|727,\d{3})", line):
                            hits.append(f"{p}:{i + 1}: {line.strip()[:120]}")
                except OSError:
                    pass
    O.append(f"\n旧値(82.1万/72.7万)の記載がある行({len(hits)}件、最大20件):")
    O += [f"- {h}" for h in hits[:20]]

    # ================= PART5
    O.append("\n## PART5 거든 の倍率(GOSE corpus.csv ÷ Run BTS、1万形態素あたり)\n")
    Gx, Rx = res["GOSE"], res["run"]
    dens = (("V2・記号除く", Gx.morph, Rx.morph), ("V2・記号含む", Gx.morph_all, Rx.morph_all))
    defs = (("반말の거든(요なし)・位置問わず(EF)", "거든_any"), ("거든요のみ・位置問わず(EF)", "거든요_any"),
            ("両方・行末のみ(v2)", "거든_final"), ("両方・位置問わず(EF)", None))
    O.append("| 数え方 | 分母 | GOSE 件数 | GOSE/1万 | Run 件数 | Run/1万 | 比 |\n|---|---|---:|---:|---:|---:|---:|")
    for lab, key in defs:
        gc = Gx.ge["거든_any"] + Gx.ge["거든요_any"] if key is None else Gx.ge[key]
        rc = Rx.ge["거든_any"] + Rx.ge["거든요_any"] if key is None else Rx.ge[key]
        for dl, gd, rd in dens:
            gp, rp = gc / gd * 1e4, rc / rd * 1e4
            O.append(f"| {lab} | {dl}(GOSE {gd} / Run {rd}) | {gc} | {gp:.2f} | {rc} | {rp:.2f} | {gp / rp:.2f} |" if rp else
                     f"| {lab} | {dl} | {gc} | {gp:.2f} | {rc} | 0 | — |")
    O.append(f"\n参考: EC としての 거든/거든요(位置問わず) GOSE {Gx.ge['거든EC_any']} / Run {Rx.ge['거든EC_any']}")
    O.append("以前の値: 반말の거든 3.92倍、거든요 GOSE 3.02・BTS 3.29(上の表の値と照合)")

    # ================= PART6
    O.append("\n## PART6 kpop_analyze.py\n")
    kp = os.path.expanduser(a.kpop)
    if os.path.exists(kp):
        st = os.stat(kp)
        O.append(f"- {kp}: {st.st_size} バイト、更新 {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")
        src = open(kp, encoding="utf-8", errors="ignore").read().splitlines()
        idx = [i for i, l in enumerate(src) if re.search(r"VV-I|VA-I|-I['\"]|endswith\(['\"]-I|split\(['\"]-['\"]\)", l)]
        if idx:
            O.append("\n該当箇所(前後2行):\n\n```")
            shown = set()
            for i in idx:
                for j in range(max(0, i - 2), min(len(src), i + 3)):
                    if j not in shown:
                        O.append(f"{j + 1}: {src[j]}")
                        shown.add(j)
                O.append("...")
            O.append("```")
        else:
            O.append("- VV-I / VA-I / -I の除外に当たるコードは見つからない(文字列検索)")
    else:
        O.append(f"- {kp}: ファイルなし")
    try:
        found = subprocess.run(["mdfind", "-name", "kpop_analyze.py"], capture_output=True, text=True).stdout.split("\n")
        found = [f for f in found if f.endswith("kpop_analyze.py")]
        O.append("\n同名ファイル:\n")
        for f in found:
            st = os.stat(f)
            O.append(f"- {f}: {st.st_size} バイト、{time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")
    except FileNotFoundError:
        O.append("\n同名ファイルの検索: 未実施(mdfind なし)")
    O.append("\n- vocab_coverage_v3_raw.pkl を作った版と同じか: 不明(確実な根拠なし)")
    return finish(a, O)


def finish(a, O):
    txt = "\n".join(O)
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
