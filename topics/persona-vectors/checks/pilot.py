# Run inside a trait's esper session after 01_extract.py's "Pilot: how big a push?" cell.
# Writes cache/checks/pilot_<TRAIT>.json: per round and α, the judge means (and their standard
# errors), how many answers hit the length cap, and the start and end of two example answers.
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)
QS = (0, 5)
rows = []
for (rnd, f), g in pilot.groupby(["round", "frac"]):
    first = g[g["sample"] == 0]
    rows.append(dict(round=rnd, frac=float(f), coeff=float(g.coeff.iloc[0]), n=len(g),
                     trait=float(g.trait_score.mean()), trait_sem=float(g.trait_score.sem()),
                     coh=float(g.coherence.mean()), coh_sem=float(g.coherence.sem()),
                     cut=float(g.truncated.mean()),
                     ex={q: dict(start=first[first.qid == q].response.iloc[0][:400],
                                 end=first[first.qid == q].response.iloc[0][-300:]) for q in QS}))
out = dict(trait=TRAIT, layer=PILOT_LAYER, positions=STEER_POS, h_len=h_len, gap_frac=gap_len / h_len,
           questions={q: eval_qs[q] for q in QS}, rows=rows)
p = Path("cache/checks")
p.mkdir(parents=True, exist_ok=True)
(p / f"pilot_{TRAIT}.json").write_text(json.dumps(out, indent=1))
log.info("exported %s's pilot (%d round × strength cells)", TRAIT, len(rows))
