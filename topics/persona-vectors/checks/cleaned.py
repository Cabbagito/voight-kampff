# Dishonest before vs after dropping its 13 extreme training answers (2026-10-30). Cache only, plain Python,
# from the topic dir: `uv run python checks/cleaned.py`. Old run: B at 18; cleaned run: A at 18 (Peter's re-pick).
#   1. cos(old vector, cleaned vector) per block;
#   2. strength curve, old vs cleaned → figures/dishonest_cleaned_strength.png;
#   3. MONITOR within-level r by answer tokens read, same 640 answers → figures/dishonest_cleaned_monitor.png.
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import savoy
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
import judge as J
import pv

trait, jd = pv.load_trait("dishonest"), J.Judge()
OLD, NEW = "old vector, B", "cleaned vector, A"
SERIES = {OLD: None, NEW: savoy.color(0)}   # the old vector is the dashed reference


def errline(ax, x, m, s=None, c=None):
    """Line with points and ±1 SE bars; c=None: the dashed reference in sn."""
    savoy.line(ax, x, m, c, baseline=c is None, points=True, yerr=s)


# 1. the vectors
scored = jd.score_df(pv.load_df("rollouts_dishonest_a63ee90e"), trait["eval_prompt"])
drop = [d["qid"] for d in trait.get("drop_extract", [])]
kept = pv.filter_rollouts(scored[~scored.qid.isin(drop)])
acts = pv.load_tensor("acts_dishonest_a63ee90e_v1_first40")
vec = lambda df: pv.persona_vectors(acts[torch.tensor(df.index.to_numpy())], df.reset_index(drop=True), match_questions=True)
old_df = pv.contrast(kept, "NEG")
new_df, dropped = pv.drop_extremes(old_df, acts)
cos = F.cosine_similarity(vec(old_df), vec(new_df), dim=-1)
print(f"dropped {len(dropped)} of {len(old_df)} ({dropped.condition.value_counts().to_dict()}) · cos per block:",
      " ".join(f"{b}:{c:.2f}" for b, c in enumerate(cos.tolist())))

# 2. strength curve
fig, axes = savoy.figure(1, 2, height=240, ytitle=["mean dishonesty (judge)", "mean coherence (judge)"],
                         xtitle="push strength (× stream length at block 18; 0 = unsteered)", sharex=True)
for lab, h in {OLD: "3c5771dd", NEW: "b7350412"}.items():
    d = jd.score_df(pv.load_df(f"dose_dishonest_{h}"), trait["eval_prompt"])
    d["frac"] = d.frac.where(d.layer >= 0, 0.0)
    g = d.groupby("frac")[["trait_score", "coherence"]].agg(["mean", "sem"])
    for ax, col in zip(axes, ["trait_score", "coherence"]):
        errline(ax, g.index, g[(col, "mean")], g[(col, "sem")], SERIES[lab])
    savoy.key(fig, lab, SERIES[lab] or savoy.T["sn"], "line" if SERIES[lab] else "base")
    print(lab, " · ".join(f"{f:g}: {r[('trait_score', 'mean')]:.0f}/{r[('coherence', 'mean')]:.0f}" for f, r in g.iterrows()))
savoy.yticks(axes[0], 0, 50); savoy.yticks(axes[1], 0, 100)
for ax in axes:
    savoy.xticks(ax, 0, 0.6)
savoy.save(fig, "figures/dishonest_cleaned_strength.png")

# 3. monitoring
READ = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
fig, ax = savoy.figure(height=260, ytitle="within-level r with judged dishonesty",
                       xtitle="answer tokens read (0 = before the answer starts)")
for lab, h in {OLD: "105c55d1", NEW: "691d06a9"}.items():
    mon = jd.score_df(pv.load_df(f"monitor_dishonest_{h}"), trait["eval_prompt"], coherence=False)
    traj = pv.load_tensor(f"traj_dishonest_{h}")   # per answer: [last prompt token, answer tokens...]
    ans = np.concatenate([t[1:].numpy() for t in traj])
    big = np.abs(ans) > 20
    r_at = lambda f: pv.monitor_correlations(mon.assign(p=[f(t) for t in traj]), proj_col="p")[1]["within_level_r"]
    rs = [r_at(lambda t: float(t[: k + 1].mean())) for k in READ]
    print(f"{lab}: median |proj| {np.median(np.abs(ans)):.1f} · {big.sum()} tokens > 20 in "
          f"{sum(bool((t[1:].abs() > 20).any()) for t in traj)} answers, median {np.median(ans[big]):.0f} · "
          f"within-level r by tokens read {dict(zip(READ, np.round(rs, 2)))} · whole answer: mean "
          f"{r_at(lambda t: float(t[1:].mean())):.2f}, median {r_at(lambda t: float(t[1:].median())):.2f}")
    errline(ax, READ, rs, c=SERIES[lab])
    savoy.key(fig, lab, SERIES[lab] or savoy.T["sn"], "line" if SERIES[lab] else "base")
    savoy.end_labels(ax, [(rs[-1], lab.split(" vector")[0], SERIES[lab] or savoy.T["sn"])])
savoy.ref(ax, 0)
ax.set_xscale("symlog", linthresh=1)
savoy.xticks(ax, 0, 256, ticks=[0, 1, 4, 16, 64, 256])
savoy.yticks(ax, 0, 0.5)
savoy.save(fig, "figures/dishonest_cleaned_monitor.png")
