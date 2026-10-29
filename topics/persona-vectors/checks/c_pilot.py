# Run inside a trait's esper session after 01_extract.py's "Steer at each layer" cell.
# How big a push does C need? C keeps the cache clean (the push only shapes the pass that picks
# the next token), so it should need more than B. At the trait's best B block (by the mean
# per-answer trait × coherence / 100), α doubling from 0.2 to 1.6 × the stream's length,
# 20 questions × 1 answer, next to B's numbers at the same block from the sweep.
# Writes cache/checks/c_pilot_<TRAIT>.json.
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)
BLOCK = {"dishonest": 18, "pirate": 24, "mystic": 18}[TRAIT]
FRACS = [0.2, 0.4, 0.8, 1.6]
cp = pv.cached_df(f"cpilot_{TAG}_{pv.key(V, BLOCK, FRACS, h_lens, 1, 'clean', STEER_SAMPLING, MAX_NEW)}",
                  lambda: pv.steering_sweep(model, tok, eval_qs, V, [BLOCK], FRACS, scale=h_lens, n_samples=1,
                                            include_baseline=False, positions="clean", device=device,
                                            **{**GEN, "batch_size": 40, **STEER_SAMPLING}))
cp = judge.score_df(cp, trait["eval_prompt"])
b = sweep[(sweep.positions == "response") & (sweep.layer == BLOCK)]


def stats(df):
    rows = []
    for f, g in df.groupby("frac"):
        rows.append(dict(frac=float(f), n=len(g), trait=float(g.trait_score.mean()), coh=float(g.coherence.mean()),
                         prod=float((g.trait_score * g.coherence / 100).mean()),
                         both=float(((g.trait_score >= 60) & (g.coherence >= 70)).mean())))
    return rows


ex = cp[cp.qid == 0].set_index("frac").response
out = dict(trait=TRAIT, block=BLOCK, C=stats(cp), B=stats(b),
           examples={str(f): dict(start=r[:300], end=r[-200:]) for f, r in ex.items()})
p = Path("cache/checks")
p.mkdir(parents=True, exist_ok=True)
(p / f"c_pilot_{TRAIT}.json").write_text(json.dumps(out, indent=1))
log.info("C at block %d, α / stream → trait/coherence: %s", BLOCK,
         " · ".join(f"{r['frac']:g}: {r['trait']:.0f}/{r['coh']:.0f}" for r in out["C"]))
