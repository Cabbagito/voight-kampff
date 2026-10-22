# Run inside a trait's esper session after 01_extract.py's "Pilot: how big a push?" cell.
# Why does the trait only show once coherence collapses? Two cheap checks at the pilot's block:
# 1. Where in the answer it breaks: re-judge the zoom round's answers cut to their first 100 and
#    250 tokens (no new generation). The unsteered answers, cut the same way, are the control.
# 2. Sampling: the paper's steered answers come from HF generate with temperature 1 and top-p 1,
#    which leaves the model's own defaults on: for Qwen2.5-7B-Instruct, top-k 20 and a repetition
#    penalty of 1.05. We sample the full distribution with no penalty. Same pushes, their sampling.
# Writes cache/checks/diagnose_<TRAIT>.json for plot.py.
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
FRACS = [0.3, 0.4]
CUTS = [100, 250]
PAPER_SAMPLING = dict(top_k=20, repetition_penalty=1.05)


def near(s, xs):
    return s.apply(lambda f: any(abs(f - x) < 1e-3 for x in xs))


z = pilot[(pilot["round"] == "zoom") & ((pilot.layer == -1) | near(pilot.frac, FRACS))]
cut = judge.score_df(pd.concat([z.assign(response=[tok.decode(tok(r, add_special_tokens=False).input_ids[:k]) for r in z.response],
                                         cut=k) for k in CUTS], ignore_index=True), trait["eval_prompt"])
cut = pd.concat([cut, z.assign(cut=500)], ignore_index=True)   # whole answers (cap 500), already judged

coeffs = [round(f * h_len, 1) for f in FRACS]
samp = pv.cached_df(f"diag_sampling_{TAG}_{pv.key(V, PILOT_LAYER, coeffs, 2, STEER_POS, MAX_NEW, PAPER_SAMPLING)}",
                    lambda: pv.steering_sweep(model, tok, eval_qs, V, [PILOT_LAYER], coeffs, n_samples=2, positions=STEER_POS,
                                              device=device, **{**GEN, "batch_size": 40, **PAPER_SAMPLING}))
samp = judge.score_df(samp, trait["eval_prompt"])
samp["frac"] = (samp.coeff / h_len).round(3)
both = pd.concat([z.assign(sampling="ours: full distribution"), samp.assign(sampling="paper's: top-k 20, repetition 1.05")],
                 ignore_index=True)


def means(df, by):
    g = df.groupby(by)[["trait_score", "coherence"]].agg(["mean", "sem"])
    return [dict(zip(by, k if isinstance(k, tuple) else (k,)), trait=r[("trait_score", "mean")], trait_sem=r[("trait_score", "sem")],
                 coh=r[("coherence", "mean")], coh_sem=r[("coherence", "sem")]) for k, r in g.iterrows()]


ex = samp[(samp.qid == 0) & (samp["sample"] == 0)].set_index("frac").response
out = dict(trait=TRAIT, layer=PILOT_LAYER, fracs=FRACS,
           cuts=means(cut.assign(frac=cut.frac.round(3)), ["frac", "cut"]),
           sampling=means(both.assign(frac=both.frac.round(3)), ["sampling", "frac"]),
           paper_sampling_examples={str(f): dict(start=r[:400], end=r[-300:]) for f, r in ex.items()},
           paper_sampling_cut=float(samp[samp.layer >= 0].truncated.mean()))
p = Path("cache/checks")
p.mkdir(parents=True, exist_ok=True)
(p / f"diagnose_{TRAIT}.json").write_text(json.dumps(out, indent=1))
s = samp.groupby("frac")[["trait_score", "coherence"]].mean()
log.info("paper's sampling at block %d, trait/coherence: %s", PILOT_LAYER,
         " · ".join(f"{f:g}: {r.trait_score:.0f}/{r.coherence:.0f}" for f, r in s.iterrows()))
