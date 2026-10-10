#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gobi.py の count 段(STEP3・4)だけを単独で動かす版。gobi_analysis/step1_2/ の抽出結果を読む。
使い方(コーパス分析フォルダで): python3 gobi_count.py
"""
import csv, os, sys
from collections import Counter, defaultdict

EXPECT_FINAL2_TOP12 = 90.1
YU_DIALECT = {"아유", "어유", "여유"}


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


class A:
    out = "gobi_analysis"

# ------------------------------------------------------------------ STEP3・4 集計(step1_2 の抽出結果を読む)
PHRASE_MIN = 300                       # 言い回しの採用基準: GOSE と BTS計 のどちらでも300件以上
METHOD_NOTES = [
    "1行=1発話。行末(記号を除いた最後の形態素。文末の 요 は語尾に含める)で判定",
    "前処理V2: 行内の [..] と (..) を消し、空白を整え、ハングルのある行だけを使う",
    "(..) は話者の注記だけでなく、初期の回ではテロップの書き方でもある。前処理V2で消しているので集計には影響しない",
    "GOSE のテロップ除外は「[..] だけでできている行」(9,212行)。用例ファイルの channel=텔롭 3,453件との照合では、"
    "発話行に残った 2,374件のうち 2,174件は行内の [..]、残り200件の多くは (..) で、どちらも前処理V2で消える",
    "まとめ方 final2: 요の有無・죠=지요・아/어/여・으/는などの有無・에요/예요→야→어・다/ㄴ다→다・나/ㄴ가→나・"
    "다고/ㄴ다고→다고・単独の요(直前が助詞/指定詞)→語尾なし・方言の 유(좋아유)→요の有無。습니다と습니까は別項目",
    "累積カバー率と上位項目の分母は「行末が終結語尾(EF)の行」。EC は別の表。EF+EC の合算表は参考で、本の数字には使わない",
    "言い回しは語尾の数え方と別に重ねて示す(言い回しとして数えた行を語尾の項目から引かない)",
]


def read_extract(o, n):
    p = os.path.join(o, "step1_2", f"extract_{n}.csv")
    with open(p, encoding="utf-8-sig", newline="") as f:
        rs = list(csv.DictReader(f))
    for r in rs:
        if r["区分"] == "語尾" and r["表層形"].split("+")[0] in YU_DIALECT:
            r["項目(final2)"] = "어"
            r["規則"] += "・要の有無(方言の유)".replace("要", "요")
    return rs


def avg_rank(c):
    items = sorted(c, key=lambda k: -c[k])
    r = {}
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and c[items[j + 1]] == c[items[i]]:
            j += 1
        for k in items[i:j + 1]:
            r[k] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** .5
    sy = sum((y - my) ** 2 for y in ys) ** .5
    return sxy / (sx * sy) if sx and sy else None


def count(a):
    import random
    o = outdir(os.path.join(a.out, "step3_4"))
    D = {n: read_extract(a.out, n) for n in ("GOSE", "run", "bomb")}
    D["BTS"] = D["run"] + D["bomb"]
    R = ["# STEP3・4 出現数・累積カバー率・重なり(まとめ方 final2)\n", "## 方法の記録\n"] + [f"- {x}" for x in METHOD_NOTES]

    def tab(rs, tag):
        c, eps, yo = Counter(), defaultdict(set), Counter()
        for r in rs:
            if r["区分"] == "語尾" and r["タグ"] == tag:
                k = r["項目(final2)"]
                c[k] += 1
                eps[k].add(r["話"])
                yo[k] += "요" in r["表層形"] or "죠" in r["表層形"] or "유" in r["表層形"]
        return c, eps, yo

    T = {}
    for n in ("GOSE", "BTS", "run", "bomb"):
        T[n] = {"EF": tab(D[n], "EF"), "EC": tab(D[n], "EC")}
    # 分母と除外
    R.append("\n## 分母と除外(行数)\n\n| | GOSE | BTS計 | Run BTS | BOMB |\n|---|---:|---:|---:|---:|")
    rows = [("発話行", lambda rs: len(rs)),
            ("行末が終結語尾(EF)の行 = 本の分母", lambda rs: sum(1 for r in rs if r["区分"] == "語尾" and r["タグ"] == "EF")),
            ("行末が連結語尾(EC)の行", lambda rs: sum(1 for r in rs if r["区分"] == "語尾" and r["タグ"] == "EC")),
            ("除外:語尾なし・その他の語尾・未区分", lambda rs: sum(1 for r in rs if r["区分"] != "語尾"))]
    for lab, fn in rows:
        R.append(f"| {lab} | " + " | ".join(str(fn(D[n])) for n in ("GOSE", "BTS", "run", "bomb")) + " |")

    def coverage(n, tag, fname, ref=False):
        c, eps, yo = T[n][tag] if not ref else (T[n]["EF"][0] + T[n]["EC"][0], None, None)
        tot = sum(c.values())
        out, cum, reach = [], 0, {}
        for i, (k, v) in enumerate(c.most_common(), 1):
            cum += v
            for p in (50, 70, 80, 90):
                if p not in reach and cum / tot * 100 >= p:
                    reach[p] = i
            out.append([i, k, v, round(v / tot, 4), round(cum / tot, 4)] + ([yo[k], len(eps[k])] if not ref else []))
        wcsv(os.path.join(o, fname), ["順位", "項目", "出現数", "割合", "累積カバー率"] + ([] if ref else ["요付き", "出現話数"]), out)
        return tot, reach, c

    R.append("\n## STEP3 終結語尾(EF)の出現数と累積カバー率(分母: 行末がEFの行)\n")
    cov = {}
    for n, fn in (("GOSE", "coverage_gose.csv"), ("BTS", "coverage_bts.csv")):
        tot, reach, c = coverage(n, "EF", fn)
        cov[n] = c
        R.append(f"- {n}: 分母 {tot} / 50% {reach.get(50)}項目、70% {reach.get(70)}項目、80% {reach.get(80)}項目、90% {reach.get(90)}項目 → {fn}")
    tot = sum(cov["GOSE"].values())
    top12 = sum(v for _, v in cov["GOSE"].most_common(12))
    R.append(f"- 確認: GOSE 上位12の占有率 {top12}/{tot} = {top12 / tot * 100:.1f}%(期待値 {EXPECT_FINAL2_TOP12}%)")
    R.append("\n| 順位 | GOSE | 件数 | 累積 | BTS計 | 件数 | 累積 |\n|---:|---|---:|---:|---|---:|---:|")
    lg, lb = cov["GOSE"].most_common(30), cov["BTS"].most_common(30)
    tg, tb = sum(cov["GOSE"].values()), sum(cov["BTS"].values())
    sg = sb = 0
    for i in range(30):
        kg, vg = lg[i] if i < len(lg) else ("", 0)
        kb, vb = lb[i] if i < len(lb) else ("", 0)
        sg += vg
        sb += vb
        R.append(f"| {i + 1} | {kg} | {vg} | {sg / tg * 100:.1f}% | {kb} | {vb} | {sb / tb * 100:.1f}% |")

    R.append("\n## 行末の連結語尾(EC)(別の表。分母: 行末がECの行)\n")
    for n, fn in (("GOSE", "ec_gose.csv"), ("BTS", "ec_bts.csv")):
        tot, reach, c = coverage(n, "EC", fn)
        R.append(f"- {n}: 分母 {tot} / 上位10: " + "、".join(f"{k} {v}" for k, v in c.most_common(10)) + f" → {fn}")
    for n, fn in (("GOSE", "ref_ef_ec_gose.csv"), ("BTS", "ref_ef_ec_bts.csv")):
        coverage(n, None, fn, ref=True)
    R.append("- 参考: EF と EC を同じ項目に合算した表 ref_ef_ec_gose.csv / ref_ef_ec_bts.csv(本の数字には使わない)")

    # 言い回し(重ねて示す)
    pc = defaultdict(lambda: {"GOSE": Counter(), "BTS": Counter(), "ends": Counter()})
    for n in ("GOSE", "BTS"):
        for r in D[n]:
            if r["区分"] == "語尾" and r["言い回し候補"]:
                x = pc[r["言い回し候補"]]
                x[n][r["タグ"]] += 1
                if n == "GOSE":
                    x["ends"][r["項目(final2)"]] += 1
    adopted = sorted([k for k, v in pc.items() if sum(v["GOSE"].values()) >= PHRASE_MIN and sum(v["BTS"].values()) >= PHRASE_MIN],
                     key=lambda k: -sum(pc[k]["GOSE"].values()))
    gef = sum(cov["GOSE"].values())
    bef = sum(cov["BTS"].values())
    prow = []
    for k in adopted:
        v = pc[k]
        prow.append([k, sum(v["GOSE"].values()), v["GOSE"]["EF"], v["GOSE"]["EC"], round(v["GOSE"]["EF"] / gef, 4),
                     sum(v["BTS"].values()), v["BTS"]["EF"], v["BTS"]["EC"], round(v["BTS"]["EF"] / bef, 4),
                     "、".join(f"{e}{c}" for e, c in v["ends"].most_common(3))])
    wcsv(os.path.join(o, "phrases.csv"), ["言い回し", "GOSE計", "GOSE 行末EF", "GOSE 行末EC", "GOSE EF行に対する割合",
                                          "BTS計", "BTS 行末EF", "BTS 行末EC", "BTS EF行に対する割合", "GOSE 後ろの語尾(上位3)"], prow)
    R.append(f"\n## 言い回し(基準: GOSE と BTS計 のどちらでも{PHRASE_MIN}件以上。{len(adopted)}種類。語尾の件数から引かない)\n")
    R.append(f"| 言い回し | GOSE計 | うち行末EF | EF行に対する割合(分母 {gef}) | BTS計 | うち行末EF | EF行に対する割合(分母 {bef}) |\n|---|---:|---:|---:|---:|---:|---:|")
    for p in prow:
        R.append(f"| {p[0]} | {p[1]} | {p[2]} | {p[4] * 100:.1f}% | {p[5]} | {p[6]} | {p[8] * 100:.1f}% |")

    # STEP4 重なり(EF)
    rg, rb = avg_rank(cov["GOSE"]), avg_rank(cov["BTS"])
    og, ob = [k for k, _ in cov["GOSE"].most_common()], [k for k, _ in cov["BTS"].most_common()]
    M = ["# STEP4 GOSE と BTS の重なり(終結語尾 EF、まとめ方 final2)\n",
         f"- 分母: GOSE EF行 {gef} / BTS計 EF行 {bef}\n", "| 上位 | 共通の項目数 |\n|---:|---:|"]
    for k in (10, 20, 30):
        M.append(f"| {k} | {len(set(og[:k]) & set(ob[:k]))} / {k} |")
    M.append(f"\n- GOSE の上位20にだけある: " + "、".join(f"{x}(BTS {int(rb[x]) if x in rb else '出現なし'}位)" for x in og[:20] if x not in ob[:20]))
    M.append(f"- BTS の上位20にだけある: " + "、".join(f"{x}(GOSE {int(rg[x]) if x in rg else '出現なし'}位)" for x in ob[:20] if x not in og[:20]))
    both = [k for k in og if k in rb]
    s_all = spearman([rg[k] for k in both], [rb[k] for k in both])
    u30 = [k for k in set(og[:30]) | set(ob[:30]) if k in rg and k in rb]
    s30 = spearman([rg[k] for k in u30], [rb[k] for k in u30])
    M.append(f"- スピアマン順位相関(両方に出てくる項目すべて、各コーパス全体での順位): {s_all:.3f}(n={len(both)})")
    M.append(f"- スピアマン順位相関(両方の上位30の和集合のうち両方に出てくる項目、各コーパス全体での順位): {s30:.3f}(n={len(u30)})")
    open(os.path.join(o, "overlap.md"), "w", encoding="utf-8").write("\n".join(M))
    R.append("\n" + "\n".join(M).replace("# STEP4", "## STEP4"))

    # 検収: 나 の EC
    na = [r for n in ("GOSE", "run", "bomb") for r in D[n] if r["区分"] == "語尾" and r["タグ"] == "EC" and r["項目(final2)"] == "나"]
    smp = random.Random(42).sample(na, min(30, len(na)))
    wcsv(os.path.join(o, "sample_check_na_ec.csv"), ["corpus", "話", "行番号", "行", "表層形", "解析",
                                                     "判定(逆接の-(으)나/それ以外/不明。空欄=未検収)"],
         [[r["corpus"], r["話"], r["行番号"], r["行"], r["表層形"], r["解析"], ""] for r in smp])
    R.append(f"\n## 検収用\n\n- sample_check_na_ec.csv: 項目「나」で行末EC の行 {len(na)}件(GOSE+Run BTS+BOMB)から30行、シード42、判定欄は空欄")
    txt = "\n".join(R)
    open(os.path.join(o, "summary_step3_4.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    count(A)
