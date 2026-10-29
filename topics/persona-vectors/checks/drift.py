# Run inside mystic's esper session after 01_extract.py's "Steer at each layer" cell (all three variants).
# Does the trait build up within an answer under each variant? Re-judges each variant's best cell
# (by trait × coherence / 100) on its first 100 and 250 tokens, next to the whole answer.
import pandas as pd
CELLS = [("all", 18, 0.333), ("response", 18, 0.333), ("clean", 24, 0.5)]
rows = []
for pos, L, f in CELLS:
    g = sweep[(sweep.positions == pos) & (sweep.layer == L) & ((sweep.frac - f).abs() < 1e-3)]
    for k in (100, 250):
        rows.append(g.assign(response=[tok.decode(tok(r, add_special_tokens=False).input_ids[:k]) for r in g.response], cut=k))
    rows.append(g.assign(cut=500))
cut = pd.concat(rows, ignore_index=True)
redo = judge.score_df(cut[cut.cut < 500], trait["eval_prompt"])
cut = pd.concat([redo, cut[cut.cut == 500]], ignore_index=True)
cut["prod"] = cut.trait_score * cut.coherence / 100
t = cut.groupby(["positions", "layer", "cut"])[["trait_score", "coherence", "prod"]].mean().round(0)
print(t.to_string())
