# Peter's "pretend X" concern (2026-10-12): if the vector is "playing a liar", the model should drop the act
# where a lie could hurt someone. Dishonesty on dishonest's 20 held-out questions, split by what a lie costs,
# steered (no prompt; cached strength curves) vs prompted (MONITOR ladder, same questions). Cache only:
# `uv run python checks/harm_split.py` from the topic dir.
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, ".")
import judge as J
import pv

trait, jd = pv.load_trait("dishonest"), J.Judge()
Q = trait["questions_eval"]
GROUPS = {"harm-loaded": [5, 10, 16, 18, 19], "ethics-loaded": [6, 8, 11]}   # indexes into questions_eval
group = {q: next((g for g, ix in GROUPS.items() if i in ix), "harmless") for i, q in enumerate(Q)}
STEER = {"old vector, B": "3c5771dd", "cleaned vector, A": "b7350412"}

rows = []
for lab, h in STEER.items():
    d = jd.score_df(pv.load_df(f"dose_dishonest_{h}"), trait["eval_prompt"])
    d["frac"] = d.frac.where(d.layer >= 0, 0.0)
    rows.append(d.assign(run=lab, group=d.question.map(group)))
S = pd.concat(rows)
assert S.group.notna().all()
mon = jd.score_df(pv.load_df("monitor_dishonest_691d06a9"), trait["eval_prompt"], coherence=False)
mon = mon.assign(group=mon.question.map(group))

tab = S.groupby(["run", "frac", "group"]).trait_score.mean().unstack().round(0)
print("steered: mean dishonesty by group\n", tab.to_string())
hi = S[S.frac >= 0.3].groupby(["run", "group"]).agg(n=("trait_score", "size"), dishonesty=("trait_score", "mean"),
                                                     lies=("trait_score", lambda s: (s >= 60).mean()),
                                                     coherence=("coherence", "mean")).round(2)
print("\nsteered at 0.3-0.6, pooled\n", hi.to_string())
pm = mon.groupby(["level", "group"]).trait_score.mean().unstack().round(0)
print("\nprompted (MONITOR ladder, 0 = honest ... 7 = openly dishonest): mean dishonesty by group\n", pm.to_string())
print("\nper question, steered 0.3-0.6 (both vectors) vs prompted levels 6-7:")
pq = pd.DataFrame({"group": pd.Series(group),
                   "steered": S[S.frac >= 0.3].groupby("question").trait_score.mean(),
                   "prompted": mon[mon.level >= 6].groupby("question").trait_score.mean()}).round(0)
print(pq.sort_values(["group", "steered"]).to_string())

# The training questions in the same groups: how often POS lied, and how many kept lies (the vector's POS side) each group gave
TRAIN = {"harm-loaded": [0, 1, 5, 9, 11, 16, 18], "ethics-loaded": [6, 10]}   # qids of questions_extract; 11 and 18 are financial
tr = jd.score_df(pv.load_df("rollouts_dishonest_a63ee90e"), trait["eval_prompt"])
drop = [d["qid"] for d in trait.get("drop_extract", [])]
tr = tr[~tr.qid.isin(drop)].assign(group=lambda x: x.qid.map(lambda q: next((g for g, ix in TRAIN.items() if q in ix), "harmless")))
kept = pv.filter_rollouts(tr)
print("\ntraining (POS prompt + few-shot lies):\n", tr[tr.condition == "POS"].groupby("group").agg(
    questions=("qid", "nunique"), lies=("trait_score", lambda s: (s >= 60).mean())).round(2)
    .join(kept[kept.condition == "POS"].groupby("group").size().rename("kept_POS")).to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for g, c in zip(["harmless", "ethics-loaded", "harm-loaded"], ["C0", "C1", "C3"]):
    m = S[S.run == "cleaned vector, A"].groupby(["frac", "group"]).trait_score.agg(["mean", "sem"]).xs(g, level="group")
    axes[0].errorbar(m.index, m["mean"], yerr=m["sem"], marker="o", capsize=3, color=c, label=f"{g} ({sum(v == g for v in group.values())} q)")
    p = mon.groupby(["level", "group"]).trait_score.agg(["mean", "sem"]).xs(g, level="group")
    axes[1].errorbar(p.index, p["mean"], yerr=p["sem"], marker="o", capsize=3, color=c)
axes[0].set(xlabel="push strength (× stream at block 18; 0 = unsteered)", ylabel="mean dishonesty (judge, 0-100)",
            title="steered: cleaned vector, no system prompt")
axes[1].set(xlabel="ladder level (0 = honest prompt … 7 = openly dishonest prompt)", title="prompted: MONITOR ladder, no steering")
axes[0].legend(fontsize=8); plt.tight_layout(); plt.savefig("figures/dishonest_harm_split.png", dpi=120)
