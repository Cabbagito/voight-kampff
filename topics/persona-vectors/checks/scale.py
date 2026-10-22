# Run inside a trait's esper session after 01_extract.py Parts 1-3 (see checks/README.md).
# Writes cache/checks/scale_<TRAIT>.json: the scale steering coefficients live on.
#
# gap: the raw matched POS − NEG difference per block, before unit-normalizing. Its length is
#   how far apart POS and NEG answers sit along the vector. The paper steers with this raw
#   vector, so its α = 1 adds one gap.
# tok_norm (dishonest session only, it's a model property): the length of the residual stream
#   per block at single answer tokens, with no system prompt, as when steering. 40 BASE answers.
import json
import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

diffs, weights = [], []
for q in lab.qid.unique():
    pos = torch.tensor(((lab.condition == "POS") & (lab.qid == q)).to_numpy())
    neg = torch.tensor(((lab.condition == "NEG") & (lab.qid == q)).to_numpy())
    if pos.any() and neg.any():
        diffs.append(acts_all[idx][pos].mean(0) - acts_all[idx][neg].mean(0))
        weights.append(pos.sum().float())
w = torch.stack(weights)
gap = ((torch.stack(diffs) * w[:, None, None]).sum(0) / w.sum()).norm(dim=-1)
out = dict(trait=TRAIT, window=REDUCE, gap=gap.tolist())

if TRAIT == "dishonest":
    base = kept[kept.condition == "BASE"].sample(40, random_state=0)
    norms = [[] for _ in range(V.shape[0])]
    with torch.no_grad():
        for h, n_prompt in pv._per_example_hidden(model, tok, base.prompt.tolist(), base.response.tolist(),
                                                  batch_size=8, device=device):
            for L, x in enumerate(h[:, n_prompt:].float().norm(dim=-1)):
                norms[L].append(x.cpu())
    norms = [torch.cat(n).numpy() for n in norms]
    out["tok_norm"] = {k: [float(np.percentile(n, p)) for n in norms] for k, p in (("p10", 10), ("median", 50), ("p90", 90))}
    log.info("residual stream length at an answer token, median: block 6 %.0f · 18 %.0f · 24 %.0f · 30 %.0f",
             *(out["tok_norm"]["median"][L] for L in (6, 18, 24, 30)))

p = Path("cache/checks")
p.mkdir(parents=True, exist_ok=True)
(p / f"scale_{TRAIT}.json").write_text(json.dumps(out))
log.info("raw POS − NEG gap length: block 6 %.1f · 18 %.1f · 24 %.1f · 30 %.1f", *(gap[L] for L in (6, 18, 24, 30)))
