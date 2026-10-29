# Run inside a trait's esper session after 01_extract.py's "Steer at different strengths" cell.
# Writes cache/checks/dose_<TRAIT>.json: at the chosen block and variant, per strength, the judge
# means (and standard errors), the mean per-answer trait × coherence / 100, the share passing
# Part 3's filter, and two example answers at the best strength.
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def stats(g):
    prod = g.trait_score * g.coherence / 100
    both = (g.trait_score >= 60) & (g.coherence >= 70)
    return dict(n=len(g), trait=float(g.trait_score.mean()), trait_sem=float(g.trait_score.sem()),
                coh=float(g.coherence.mean()), coh_sem=float(g.coherence.sem()),
                prod=float(prod.mean()), prod_sem=float(prod.sem()), both=float(both.mean()), cut=float(g.truncated.mean()))


rows = [dict(frac=float(f), **stats(g)) for f, g in dose[dose.layer >= 0].groupby("frac")]
best = max(rows, key=lambda r: r["prod"])["frac"]
b = dose[(dose.layer >= 0) & ((dose.frac - best).abs() < 1e-6)].assign(prod=dose.trait_score * dose.coherence / 100)
ex = b.sort_values("prod").iloc[[len(b) // 2, -1]]   # a typical answer and the best one
out = dict(trait=TRAIT, layer=LAYER, positions=STEER_POS, h_len=float(h_lens[LAYER]), unsteered=stats(dose[dose.layer == -1]),
           rows=rows, best_frac=best,
           examples=[dict(question=r.question, trait=float(r.trait_score), coh=float(r.coherence), start=r.response[:400],
                          end=r.response[-200:]) for _, r in ex.iterrows()])
p = Path("cache/checks")
p.mkdir(parents=True, exist_ok=True)
(p / f"dose_{TRAIT}.json").write_text(json.dumps(out, indent=1))
log.info("exported %s's strength curve (best at %g × stream)", TRAIT, best)
