# PART0 診断: 消えた clean_text の候補を試し、過去の total_tokens / 잖아・거든件数に合うものを探す
import json, os, re, sys
from kiwipiepy import Kiwi
D = sys.argv[1] if len(sys.argv) > 1 else "seventeen_analysis"
PAST = {  # concentration_ranking.csv (GOSE): ep -> (word, count, total_tokens)
 "h2tvKytryQU":("거든",11,6463),"Z4guem5XzCw":("거든",10,5893),"aOmz9GV_Uxk":("거든",16,10090),
 "Ujkg9TO1-TQ":("거든",9,5758),"_7KUIRQUAIQ":("거든",9,6016),"_O0UWg3rzx4":("잖아",37,6470),
 "PLUfUXEkPm4":("잖아",26,4743),"UIlZKtRyog0":("잖아",31,5776),"UO2KBxvkodo":("잖아",29,5405),
 "Sd9HkHMC7Wo":("잖아",35,6616)}
ws = lambda t: re.sub(r"\s+", " ", t).strip()
br = lambda t: re.sub(r"\[[^\]]*\]", " ", t)
pa = lambda t: re.sub(r"\([^)]*\)", " ", t)
sym = lambda t: re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9\s?!.,]", " ", t)
allp = lambda t: re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9\s]", " ", t)
hang = lambda t: re.sub(r"[^가-힣\s]", " ", t)
V = {"V0 空白のみ": lambda t: ws(t), "V1 [..]削除": lambda t: ws(br(t)),
     "V2 [..](..)削除": lambda t: ws(pa(br(t))), "V3 V2+記号削除(?!.,残す)": lambda t: ws(sym(pa(br(t)))),
     "V4 V2+記号全削除": lambda t: ws(allp(pa(br(t)))), "V5 V2+ハングル以外全削除": lambda t: ws(hang(pa(br(t))))}
k = Kiwi()
raw = {}
for ep in PAST:
    segs = json.load(open(os.path.join(D, ep + "_utterances.json"), encoding="utf-8"))
    segs = segs if isinstance(segs, list) else segs.get("segments", segs)
    raw[ep] = [str(s.get("text", "")) if isinstance(s, dict) else str(s) for s in segs]
print("variant | 거든計(過去55) | 잖아計(過去158) | tokens差の平均 | tokens一致話数(±1%)")
for name, f in V.items():
    cs = {"거든": 0, "잖아": 0}; diffs = []; hit = 0
    for ep, (w, c0, t0) in PAST.items():
        tot = n = 0
        for t in raw[ep]:
            t = f(t.strip())
            if not re.search(r"[가-힣]", t): continue
            toks = k.tokenize(t); tot += len(toks)
            n += sum(1 for x in toks if x.tag == "EF" and x.form == w)
        cs[w] += n; d = (tot - t0) / t0; diffs.append(d); hit += abs(d) <= 0.01
    print(f"{name} | {cs['거든']} | {cs['잖아']} | {sum(diffs)/len(diffs)*100:+.1f}% | {hit}/10")
