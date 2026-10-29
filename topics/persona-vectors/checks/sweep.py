# Run inside a trait's esper session after 01_extract.py's "Steer at each layer" cell.
# Writes cache/checks/sweep_<TRAIT>.json: per variant (A / B / C), strength and block, the judge
# means (and their standard errors), the mean per-answer trait × coherence / 100, the share of
# answers with trait >= 60 and coherence >= 70 (Part 3's filter), and the share that hit the length
# cap; the unsteered baseline; and the opening and ending of one answer per cell for two questions.
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)
QS = (0, 5)


def stats(g):
    both = (g.trait_score >= 60) & (g.coherence >= 70)   # Part 3's filter: shows the trait, and coherent
    prod = g.trait_score * g.coherence / 100
    return dict(n=len(g), trait=float(g.trait_score.mean()), trait_sem=float(g.trait_score.sem()),
                coh=float(g.coherence.mean()), coh_sem=float(g.coherence.sem()), cut=float(g.truncated.mean()),
                both=float(both.mean()), both_sem=float(both.sem()), prod=float(prod.mean()), prod_sem=float(prod.sem()))


steered = sweep[sweep.layer >= 0]
cells = []
for (pos, f, L), g in steered.groupby(["positions", "frac", "layer"]):
    first = g[g["sample"] == 0]
    cells.append(dict(positions=pos, frac=float(f), layer=int(L), alpha=float(g.coeff.iloc[0]), **stats(g),
                      ex={q: dict(start=first[first.qid == q].response.iloc[0][:300],
                                  end=first[first.qid == q].response.iloc[0][-200:]) for q in QS}))
out = dict(trait=TRAIT, sampling=STEER_SAMPLING, sweeps={pos: dict(fracs=f, layers=l) for pos, (f, l) in SWEEPS.items() if f},
           h_lens=[float(x) for x in h_lens], unsteered=stats(sweep[sweep.layer == -1]),
           questions={q: eval_qs[q] for q in QS}, cells=cells)
p = Path("cache/checks")
p.mkdir(parents=True, exist_ok=True)
(p / f"sweep_{TRAIT}.json").write_text(json.dumps(out, indent=1))
log.info("exported %s's layer sweeps (%d variant × strength × block cells)", TRAIT, len(cells))
