#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GOSE 語尾・助詞探索 第1便 PART1〜4(観察のみ)

前処理(PART0で過去集計と一致を確認した V2):
  行内の [..] と (..) を削除し空白を正規化。結果にハングルが無い行は除外(テロップ扱い)。
  話者ターンの集約はしない(残った全行が母集団)。

使い方:
  python3 part1_4_endings.py --corpus corpus.csv --list list_gose_complete.csv --out out_endings
出力(--out 内): (a)〜(h) の CSV、kiwi_suspects.csv、summary.md。画面には summary.md を出す。
"""
import argparse
import csv
import os
import random
import re
from collections import Counter, defaultdict

import kiwipiepy
from kiwipiepy import Kiwi

SEED = 42
BR = re.compile(r"\[[^\]]*\]")
PA = re.compile(r"\([^)]*\)")
HANGUL = re.compile(r"[가-힣]")
J_TAGS = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"]
MEMBERS = ["에스쿱스", "쿱스", "승철", "정한", "조슈아", "슈아", "지수", "문준휘", "준휘", "준",
           "호시", "순영", "원우", "우지", "지훈", "디에잇", "명호", "민규", "도겸", "석민",
           "승관", "부승관", "버논", "한솔", "디노", "찬"]


def clean(t):
    return re.sub(r"\s+", " ", PA.sub(" ", BR.sub(" ", t))).strip()


def to_sec(t):
    try:
        p = [int(x) for x in str(t).strip().split(":")]
    except ValueError:
        return None
    s = 0
    for x in p:
        s = s * 60 + x
    return s


def base_tag(tag):
    return tag.split("-")[0]  # VV-R / VA-I などの不規則印を外す


JAMO = str.maketrans({"\u11ab": "ㄴ", "\u11af": "ㄹ", "\u11b8": "ㅂ", "\u11b7": "ㅁ", "\u11bb": "ㅆ"})
KEEP_LEX = {"같", "있", "없", "하", "되", "싶"}  # 並びで形を残す用言(文法化した構文で使うもの)


def nf(form):
    return form.translate(JAMO)  # kiwi の終声字母(ᆫ 等)を互換字母(ㄴ 等)に統一


def strip_yo(form):
    """語尾の 요 を外して見出し形にする。죠 は 지+요 として扱う"""
    form = nf(form)
    if form == "죠":
        return "지", True
    if form.endswith("요") and len(form) > 1:
        return form[:-1], True
    return form, False


def norm_tok(t):
    tag = base_tag(t.tag)
    f = nf(t.form)
    # 内容語はタグだけに置き換える(学習単位に近い形にするため)。文法化した用言・依存名詞は形を残す
    if tag in ("NNG", "NNP", "NP", "NR", "MAG", "MAJ", "MM", "IC", "XR", "SL", "SN", "SH") or \
            (tag in ("VV", "VA") and f not in KEEP_LEX):
        return tag
    if tag == "ETM":
        f = {"을": "ㄹ", "은": "ㄴ"}.get(f, f)
    if tag == "NNB" and f == "것":
        f = "거"
    if tag.startswith("E"):
        f = strip_yo(f)[0]
    return f"{f}/{tag}"


class Item:
    __slots__ = ("n", "yo", "eps", "hits")

    def __init__(self):
        self.n = 0
        self.yo = 0
        self.eps = Counter()
        self.hits = []  # 行の通し番号


def add(d, key, li, ep, yo):
    it = d.get(key)
    if it is None:
        it = d[key] = Item()
    it.n += 1
    it.yo += yo
    it.eps[ep] += 1
    it.hits.append(li)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--list")
    ap.add_argument("--out", default="out_endings")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rnd = random.Random(SEED)

    # ------------------------------------------------ 読み込み・前処理
    lines = []  # dict(ep, vid, time, row, raw, text)
    n_rows = n_drop = 0
    with open(a.corpus, encoding="utf-8-sig", newline="") as f:
        for i, r in enumerate(csv.DictReader(f), 1):
            n_rows += 1
            raw = (r.get("text") or "").strip()
            t = clean(raw)
            if not HANGUL.search(t):
                n_drop += 1
                continue
            lines.append({"ep": r.get("episode", ""), "vid": r.get("video_id", ""),
                          "time": r.get("time", ""), "row": i, "raw": raw, "text": t})
    kiwi = Kiwi()
    toks_all = list(kiwi.tokenize([l["text"] for l in lines]))
    N = len(lines)
    # 同じ話の前後行
    for i, l in enumerate(lines):
        l["prev"] = lines[i - 1]["text"] if i > 0 and lines[i - 1]["ep"] == l["ep"] else ""
        l["next"] = lines[i + 1]["text"] if i + 1 < N and lines[i + 1]["ep"] == l["ep"] else ""
        l["next_time"] = lines[i + 1]["time"] if l["next"] else ""

    # ------------------------------------------------ 集計
    final_e, internal_ec, parts, tails, noend = {}, {}, {}, {}, {}
    part_tags = defaultdict(Counter)
    noend_tag = Counter()
    final_kind = Counter()
    n_morph = 0
    line_final = [None] * N  # (key, tag, yo)
    suspects = []
    list_words = set()
    list_col = None
    if a.list:
        with open(a.list, encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            cands = [c for c in rd.fieldnames if c and c.strip().lower() in
                     ("lemma", "word", "語", "単語", "단어", "표제어", "見出し語", "base", "見出し")]
            list_col = cands[0] if cands else rd.fieldnames[0]
            for r in rd:
                w = (r.get(list_col) or "").strip()
                if w:
                    list_words.add(w)
    line_lemmas = [set() for _ in range(N)]

    for li, (l, toks) in enumerate(zip(lines, toks_all)):
        ep = l["ep"]
        core = [t for t in toks if not t.tag.startswith("S")]
        n_morph += len(core)
        # 語彙形(PART3用)
        for k, t in enumerate(core):
            tg = base_tag(t.tag)
            if tg in ("VV", "VA", "VX"):
                line_lemmas[li].add(t.form + "다")
            elif tg in ("XSV", "XSA") and k > 0 and core[k - 1].tag.startswith("N"):
                line_lemmas[li].add(core[k - 1].form + t.form + "다")
            elif tg[0] in "NM" or tg == "IC":
                line_lemmas[li].add(t.form)
        if not core:
            continue
        # 行末(記号を除く)。末尾の 요/JX は直前要素の「요付き」として扱う
        end = len(core) - 1
        yo_tail = False
        if end > 0 and core[end].form == "요" and core[end].tag in ("JX", "MM"):
            yo_tail = True
            if core[end].tag == "MM":
                suspects.append(("行末の 요 が MM", l, " ".join(f"{t.form}/{t.tag}" for t in core)))
            end -= 1
        last = core[end]
        ltag = base_tag(last.tag)
        if ltag.startswith("E"):
            f, yo_in = strip_yo(last.form)
            yo = yo_in or yo_tail
            kind = ltag if ltag in ("EF", "EC") else "その他E"
            add(final_e, (f, ltag), li, ep, yo)
            line_final[li] = (f, ltag, yo)
            final_kind[kind] += 1
        else:
            key = nf(last.form) if ltag not in ("VV", "VA", "VX") else last.form + "다"
            add(noend, (key, ltag), li, ep, yo_tail)
            noend_tag[ltag] += 1
            final_kind["語尾なし"] += 1
        # 行末 2〜4 形態素の並び(末尾が語尾の行のみ)
        if ltag.startswith("E"):
            seq = core[: end + 1]
            for n in (2, 3, 4):
                if len(seq) >= n:
                    add(tails, " + ".join(norm_tok(t) for t in seq[-n:]), li, ep, line_final[li][2])
        # 行の途中の EC / 助詞
        for k, t in enumerate(core):
            tg = base_tag(t.tag)
            nxt_yo = k + 1 < len(core) and core[k + 1].form == "요" and core[k + 1].tag == "JX"
            if tg == "EC" and k < end:
                f, yo_in = strip_yo(t.form)
                add(internal_ec, (f, "EC"), li, ep, yo_in or nxt_yo)
            if tg.startswith("J") and not (t.form == "요" and tg == "JX" and k == len(core) - 1 and yo_tail):
                add(parts, (nf(t.form), "J"), li, ep, nxt_yo)
                part_tags[nf(t.form)][tg] += 1
        # kiwi 誤解析の疑い
        for k in range(len(core) - 2):
            if core[k].tag == "NNP" and core[k + 1].tag == "VCP" and core[k + 2].form in ("야", "아") \
                    and core[k + 2].tag == "EF":
                suspects.append(("人名+이/VCP+야/EF(呼びかけの可能性)", l,
                                 " ".join(f"{t.form}/{t.tag}" for t in core)))
                break
        if len(HANGUL.findall(l["text"])) <= 2 and ltag.startswith("E"):
            suspects.append(("短い断片で行末が語尾", l, " ".join(f"{t.form}/{t.tag}" for t in core)))
        for m in MEMBERS:
            for mo in re.finditer(m, l["text"]):
                s, e = mo.start(), mo.end()
                ov = [t for t in toks if t.start < e and t.start + t.len > s]
                if not (len(ov) == 1 and ov[0].tag.startswith("N")):  # 名前が1つの名詞に収まっていれば正常
                    suspects.append(("名前の切れ端に語尾/助詞/接辞", l,
                                     " ".join(f"{t.form}/{t.tag}" for t in toks)))
                    break

    denom = n_morph

    # ------------------------------------------------ 書き出し共通
    def kwic(hits, k=3):
        pick = hits if len(hits) <= k else rnd.sample(hits, k)
        out = []
        for li in sorted(pick):
            l = lines[li]
            out += [f"{l['ep']}#{l['row']}", l["prev"], l["text"], l["next"]]
        out += [""] * (4 * k - len(out))
        return out

    KW_HEAD = [f"KWIC{i}_{c}" for i in (1, 2, 3) for c in ("話#行", "前行", "行", "次行")]
    COMMON = ["形", "タグ", "件数", "1万形態素あたり", "요付き件数", "요なし件数", "반말比率",
              "出現話数", "上位5話集中率"]

    def common_vals(form, tag, it):
        top5 = sum(c for _, c in it.eps.most_common(5))
        return [form, tag, it.n, round(it.n / denom * 10000, 2), it.yo, it.n - it.yo,
                round((it.n - it.yo) / it.n, 3), len(it.eps), round(top5 / it.n, 3)]

    def write(name, head, rows):
        with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(rows)

    def ranked(d):
        return sorted(d.items(), key=lambda kv: -kv[1].n)

    # (a)
    fin = ranked(final_e)
    write("endings_line_final.csv", COMMON + ["区分", "文末で使われた連結語尾"] + KW_HEAD,
          [common_vals(f, t, it) + [t if t in ("EF", "EC") else "その他", "○" if t == "EC" else ""]
           + kwic(it.hits) for (f, t), it in fin])
    # (a) PART2: 行末ECの各行
    ec_rows = []
    ec_by_item = defaultdict(list)
    for li, lf in enumerate(line_final):
        if lf and lf[1] == "EC":
            l = lines[li]
            s0, s1 = to_sec(l["time"]), to_sec(l["next_time"])
            gap = s1 - s0 if s0 is not None and s1 is not None else ""
            row = [l["ep"], l["row"], l["time"], lf[0], "요" if lf[2] else "", l["text"],
                   l["next"][:30], l["next_time"], gap]
            ec_rows.append(row)
            ec_by_item[lf[0]].append(row)
    EC_HEAD = ["話", "行番号", "時刻", "語尾", "요", "行", "次行の冒頭", "次行時刻", "間隔秒"]
    write("endings_line_final_EC_lines.csv", EC_HEAD, ec_rows)
    top_ec = [f for (f, t), _ in fin if t == "EC"][:10]
    smp = []
    for f in top_ec:
        rs = ec_by_item[f]
        smp += rnd.sample(rs, min(10, len(rs)))
    write("sample_check_line_final_EC.csv", EC_HEAD + ["判定(文末/次行に続く/不明)"], [r + [""] for r in smp])
    # (b)
    write("endings_EC_internal.csv", COMMON + KW_HEAD,
          [common_vals(f, t, it) + kwic(it.hits) for (f, t), it in ranked(internal_ec)])
    # (c)
    write("particles_J.csv", COMMON + J_TAGS + KW_HEAD,
          [common_vals(f, "/".join(t for t, _ in part_tags[f].most_common()), it)
           + [part_tags[f][j] for j in J_TAGS] + kwic(it.hits) for (f, _), it in ranked(parts)])
    # (d)
    tl = ranked(tails)[:200]
    write("endings_tail.csv", ["並び", "形態素数"] + COMMON[2:] + KW_HEAD,
          [[k, k.count("+") + 1] + common_vals(k, "", it)[2:] + kwic(it.hits) for k, it in tl])
    # (e)
    ne = ranked(noend)
    e_rows = []
    for tag, cnt in noend_tag.most_common():
        e_rows.append(["(全体)", tag, cnt, round(cnt / denom * 10000, 2)] + [""] * (len(COMMON) - 4) + [""] * 12)
        for (w, t), it in [x for x in ne if x[0][1] == tag][:10]:
            e_rows.append(common_vals(w, t, it) + kwic(it.hits))
    write("line_final_no_ending.csv", COMMON + KW_HEAD, e_rows)
    # (f) PART3
    top50 = fin[:50]
    co_rows = []
    for (f, t), it in top50:
        c = Counter()
        for li in it.hits:
            for w in line_lemmas[li] & list_words:
                c[w] += 1
        co_rows.append([f, t, it.n] + [f"{w}:{n}" for w, n in c.most_common(10)])
    write("ending_word_cooc.csv", ["形", "タグ", "件数"] + [f"共起{i}" for i in range(1, 11)], co_rows)
    # (g) PART4
    g_rows = []
    for (f, t), it in fin[:20]:
        for li in sorted(rnd.sample(it.hits, min(10, len(it.hits)))):
            l = lines[li]
            g_rows.append([f, t, l["ep"], l["row"], l["time"], l["prev"], l["text"], l["next"], ""])
    write("sample_check_endings.csv", ["形", "タグ", "話", "行番号", "時刻", "前行", "行", "次行",
                                       "判定(空欄=未検収)"], g_rows)
    # (h)
    write("sample_audio_check_30.csv", ["話", "video_id", "時刻", "行番号", "原文", "前処理後", "照合メモ"],
          [[lines[i]["ep"], lines[i]["vid"], lines[i]["time"], lines[i]["row"], lines[i]["raw"],
            lines[i]["text"], ""] for i in sorted(rnd.sample(range(N), min(30, N)))])
    # 誤解析の疑い
    sc = Counter(k for k, _, _ in suspects)
    write("kiwi_suspects.csv", ["種別", "話", "行番号", "行", "解析"],
          [[k, l["ep"], l["row"], l["text"], an] for k, l, an in suspects])
    multi = []
    tag_by_form = defaultdict(Counter)
    for (f, t), it in final_e.items():
        tag_by_form[f][t] += it.n
    for (f, t), it in internal_ec.items():
        tag_by_form[f]["EC(行中)"] += it.n
    for f, c in tag_by_form.items():
        if len(c) > 1 and sum(c.values()) >= 30:
            multi.append((sum(c.values()), f, dict(c)))
    multi.sort(reverse=True)

    # ------------------------------------------------ 要約
    S = [f"# PART1〜4 要約\n\nkiwipiepy {kiwipiepy.__version__} / 前処理 V2\n",
         f"- corpus 行数 {n_rows} / 除外(V2後ハングル無し) {n_drop} / 母集団 {N} 行",
         f"- 形態素数(記号除く・1万あたりの分母) {denom}",
         f"- list 列: {list_col} / 語数 {len(list_words)}" if a.list else "- list 未指定"]
    tot = sum(final_kind.values())
    S.append("\n## 行末の内訳\n")
    for k in ("EF", "EC", "その他E", "語尾なし"):
        S.append(f"- {k}: {final_kind[k]} ({final_kind[k] / tot * 100:.1f}%)")

    def table(title, rows, head):
        S.append(f"\n## {title}\n")
        S.append("| " + " | ".join(head) + " |")
        S.append("|" + "---|" * len(head))
        for r in rows:
            S.append("| " + " | ".join(str(x) for x in r) + " |")

    H = ["形", "タグ", "件数", "1万", "요付", "요なし", "반말比", "話数", "上位5集中"]
    table("(a) 行末の語尾 上位20", [common_vals(f, t, it) for (f, t), it in fin[:20]], H)
    table("(b) 行中の連結語尾 上位20", [common_vals(f, t, it) for (f, t), it in ranked(internal_ec)[:20]], H)
    table("(c) 助詞 上位20", [common_vals(f, "/".join(f"{t}{n}" for t, n in part_tags[f].most_common(3)), it)
                            for (f, _), it in ranked(parts)[:20]], H)
    table("(d) 行末の並び 上位20", [[k, it.n, it.yo, len(it.eps)] for k, it in tl[:20]],
          ["並び", "件数", "요付", "話数"])
    table("(e) 語尾なしで終わる行 末尾品詞 上位", [[t, c, f"{c / tot * 100:.1f}%"] for t, c in noend_tag.most_common(12)],
          ["品詞", "件数", "行末比"])
    S.append("\n## kiwi 誤解析の疑い\n")
    for k, c in sc.items():
        S.append(f"- {k}: {c} 行(kiwi_suspects.csv)")
    for k in sc:
        ex = [x for x in suspects if x[0] == k][:5]
        for _, l, an in ex:
            S.append(f"  - {l['ep']}#{l['row']} 「{l['text'][:40]}」 → {an[:80]}")
    S.append("- 同形で複数タグに出る形(計30件以上、上位15):")
    for n, f, c in multi[:15]:
        S.append(f"  - {f}: {c}")
    S.append("\n## 正規化の注記\n\n- 語尾末尾の 요 を外して見出し形に統合(죠 は 지+요 扱い)。行末の 요/JX は直前要素の 요付き。"
             "\n- 並び: 内容語はタグのみ(같/있/없/하/되/싶 は形を残す)。ETM 을→ㄹ・은→ㄴ、NNB 것→거、不規則印(-R/-I 等)を除去。"
             "\n- kiwi の終声字母(ᆫ ᆯ ᆸ ᆷ ᆻ)は互換字母(ㄴ ㄹ ㅂ ㅁ ㅆ)に統一。\n- 1万形態素あたりの分母は記号(S*)を除いた形態素数。")
    txt = "\n".join(S)
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
