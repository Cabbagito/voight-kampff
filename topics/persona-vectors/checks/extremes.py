# Did extreme values shape the persona vector? (Peter, 2026-10-29, after the MONITOR spikes)
# Run inside a trait's esper session after 01_extract.py's "Build the persona vectors" cell
# (uses acts_all, idx, lab, TRAIT, pv). Per block, on the training answers' averaged activations:
#   1. outlier answers: any dimension more than Z robust SDs from its own median (pv.extremes);
#   2. A (a few answers dragged the vector): rebuild without them, compare;
#      B (a broad shift): the vector barely changes.
import logging

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)
Z, SHOW = 10.0, 18

X = acts_all[idx]                                       # (n_answers, n_blocks, d): the answers V is built from
pos = torch.tensor((lab.condition == "POS").to_numpy())
raw = pv.persona_vectors(X, lab, match_questions=True, normalize=False)
rows = []
for b in range(raw.shape[0]):
    ext = pv.extremes(X[:, b], Z)
    bad = ext.any(1)
    dims = ext.sum(0).nonzero().flatten()
    clean = pv.persona_vectors(X[~bad][:, b : b + 1], lab[(~bad).numpy()].reset_index(drop=True),
                               match_questions=True, normalize=False)[0]
    s = raw[b] ** 2 / (raw[b] ** 2).sum()
    rows.append(dict(block=b, n_bad=int(bad.sum()), bad_pos=int(bad[pos].sum()), dims=dims.tolist()[:6],
                     share=float(s[dims].sum()), share_clean=float((clean[dims] ** 2).sum() / (clean ** 2).sum()),
                     cos_clean=float(F.cosine_similarity(clean, raw[b], dim=0))))
    if b == SHOW:
        top = s.topk(3).indices.tolist()
        log.info("block %d, the vector's biggest dims: %s", b, " · ".join(f"dim {k} {100 * s[k]:.0f}%" for k in top))
        for k in dims.tolist()[:4]:
            c = X[:, b, k]
            log.info("block %d dim %d: typical %.1f, outliers %s · POS − NEG mean gap %.1f with them, %.1f without",
                     b, k, float(c.median()), [round(float(v)) for v in c[ext[:, k]]],
                     float(c[pos].mean() - c[~pos].mean()), float(c[pos & ~bad].mean() - c[~pos & ~bad].mean()))
        bq = lab[bad.numpy()]
        log.info("block %d outlier answers: %s · questions %s", b, bq.condition.value_counts().to_dict(),
                 bq.qid.value_counts().to_dict())

R = pd.DataFrame(rows)
display(R.round(3))
r = R.loc[SHOW]
log.info("block %d: %d of %d answers are outliers (%d POS of %d) · %.0f%% of the vector sits in their dims → %.0f%% without them · "
         "cos(rebuilt, original) %.2f", SHOW, r.n_bad, len(X), r.bad_pos, int(pos.sum()), 100 * r.share, 100 * r.share_clean, r.cos_clean)

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(R.block, R.cos_clean, "o-", color="C3")
ax.axvspan(15.5, 19.5, color="gray", alpha=.15, label="blocks 16–19 (Part 3's quirk)")
ax.set(xlabel="block (layer, 0-35)", ylabel="cosine: rebuilt vs original",
       title=f"{TRAIT}: the vector rebuilt without its outlier answers", ylim=(0, 1.02))
ax.legend(fontsize=8, loc="lower left"); plt.tight_layout()
plt.savefig(f"figures/{TRAIT}_extremes.png", dpi=120)
