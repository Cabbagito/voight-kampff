# Checks behind Exp 1's Part 3 decisions

These scripts run **inside a trait's esper session**, after `01_extract.py` has been run
through Part 3 there. They read its live variables (`TRAIT`, `acts_by`, `kept`, `main`,
`lab`, `idx`, `scored`, `cos_tok`, `cos_half`, `cos_nb`, `pv`). Each one either writes a
figure straight into `../figures/` or writes its numbers to `../cache/checks/<name>_<trait>.json`
for `plot.py` to draw.

```
# per trait session (dishonest, pirate, mystic):
uv run esper exec - -s pirate --title "export for the cross-trait plots" < topics/persona-vectors/checks/export.py
# then, once all three traits are exported, plain Python:
uv run python topics/persona-vectors/checks/plot.py
# per trait session, for Part 4's steering scale (dishonest also measures the residual stream):
uv run esper exec - -s pirate --title "scale for α" < topics/persona-vectors/checks/scale.py
# dishonest only:
uv run esper exec - -s dishonest --title "block 16" < topics/persona-vectors/checks/block16.py
```

| script | what it settles | figures |
|---|---|---|
| `export.py` | the numbers for every cross-trait plot: scores, whole answer vs first 40 tokens (cosine, separation, length), pooled vs matched, POS − NEG vs POS − BASE | via `plot.py` |
| `plot.py` | draws the cross-trait figures | `scores_three_traits`, `first40_three_traits`, `window_separation_length`, `pooled_vs_matched_three_traits`, `pos_neg_vs_pos_base` |
| `pilot.py` | exports the strength pilot at block 24 (run after the pilot cell) | `pilot_strength` (via `plot.py`) |
| `diagnose.py` | why the pilot shows no window: re-judges answers cut to their first 100 / 250 tokens; repeats two strengths with the paper's sampling (top-k 20, repetition penalty 1.05) | numbers in `cache/checks/diagnose_<trait>.json` |
| `scale.py` | the units of the steering coefficient α: residual stream length per block, each trait's raw POS − NEG gap | `steering_scale` (via `plot.py`) |
| `block16.py` | why every curve breaks at block 16 (dishonest; compares other traits and plain inputs) | `dishonest_stability_separation`, `dishonest_block16`, `block16_any_input` |
