# Run inside a trait's esper session after 01_extract.py Parts 1-3 (see checks/README.md).
# Writes cache/checks/export_<TRAIT>.json with everything the cross-trait plots need.
#
# Separation: build the vector from a random half of the kept answers, score the other half;
# within each question, Cohen's d between POS and NEG projections (gap between the group
# means over their typical spread), weighted by the question's number of POS answers.
# 20 random splits. Length direction: NEG answers only, within each question, the longer
# half's mean activation minus the shorter half's, averaged over questions.
import json
import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)
F = torch.nn.functional.cosine_similarity
cond, qid = lab.condition.to_numpy(), lab.qid.to_numpy()
WINDOW = REDUCE   # this trait's chosen token window


def within_d(P, test):
    ds, ws = [], []
    for q in np.unique(qid):
        pos, neg = test & (qid == q) & (cond == "POS"), test & (qid == q) & (cond == "NEG")
        if pos.sum() >= 2 and neg.sum() >= 2:
            pp, nn = P[torch.from_numpy(pos)], P[torch.from_numpy(neg)]
            ds.append((pp.mean(0) - nn.mean(0)) / torch.sqrt((pp.var(0) + nn.var(0)) / 2))
            ws.append(float(pos.sum()))
    w = torch.tensor(ws)
    return (torch.stack(ds) * w[:, None]).sum(0) / w.sum()


def separation(a, match):
    runs = []
    for seed in range(20):
        h1 = lab.groupby(["condition", "qid"]).sample(frac=.5, random_state=seed).index.to_numpy()
        m = np.zeros(len(lab), bool)
        m[h1] = True
        v1 = pv.persona_vectors(a[torch.from_numpy(m)], lab[m].reset_index(drop=True), match_questions=match)
        runs.append(within_d(torch.einsum("nld,ld->nl", a, v1), ~m))
    S = torch.stack(runs)
    return dict(mean=S.mean(0).tolist(), lo=S.min(0).values.tolist(), hi=S.max(0).values.tolist())


def length_cos(A, match):
    neg_idx = kept[kept.condition == "NEG"].index
    ds = []
    for _, g in kept.loc[neg_idx].groupby("qid"):
        med = g.n_tokens.median()
        lo, hi = g[g.n_tokens < med].index, g[g.n_tokens > med].index
        if len(lo) >= 3 and len(hi) >= 3:
            ds.append(A[torch.tensor(hi.to_numpy())].mean(0) - A[torch.tensor(lo.to_numpy())].mean(0))
    if not ds:
        return None
    return F(pv.persona_vectors(A[idx], lab, match_questions=match), torch.stack(ds).mean(0), dim=-1).tolist()


out = dict(trait=TRAIT, window=WINDOW,
           scores={c: dict(trait=g.trait_score.dropna().tolist(), coherence=g.coherence.dropna().tolist())
                   for c, g in scored.groupby("condition")},
           lengths={c: dict(median=float(g.n_tokens.median()), cut=float(g.truncated.mean())) for c, g in kept.groupby("condition")},
           cos_full_first40=cos_tok.tolist(), noise=cos_half.tolist(),
           window_check={red: dict(sep=separation(A[idx], False), len_cos=length_cos(A, False)) for red, A in acts_by.items()},
           match_check={k: separation(acts_by[WINDOW][idx], m) for k, m in (("pooled", False), ("matched", True))},
           cos_pooled_matched=F(pv.persona_vectors(acts_by[WINDOW][idx], lab),
                                pv.persona_vectors(acts_by[WINDOW][idx], lab, match_questions=True), dim=-1).tolist(),
           cos_neg_base=cos_nb.tolist())
p = Path("cache/checks")
p.mkdir(parents=True, exist_ok=True)
(p / f"export_{TRAIT}.json").write_text(json.dumps(out))
log.info("exported %s's numbers for the cross-trait plots (%s window)", TRAIT, WINDOW)
