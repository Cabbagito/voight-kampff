# Experiment: <section number> <name>

> Template. Copy this directory to `experiments/<section>/` when starting
> a new section.

## What
One paragraph: what technique is this experiment exercising, and what
question does this specific run try to answer (vs. just replicating)?

## Why
Why am I doing this — what would I learn from a clean result vs. a
surprise? Tie to the broader curriculum goal.

## How to run
```bash
uv run python -m experiments.<section>.src.main  # or jupyter notebook
```

## Status
- [ ] ARENA markdown fetched (`context/arena.md`)
- [ ] Paper(s) downloaded to `context/` and read
- [ ] `context/brief.md` written
- [ ] Discussed / quizzed
- [ ] Stubs scaffolded
- [ ] `predictions.md` written
- [ ] Substance filled in
- [ ] Ran, compared to predictions
- [ ] Findings noted in `results/`

## Files
- `context/arena.md` — ARENA section markdown (gitignored content; checked-in skeleton ok)
- `context/brief.md` — my 1-page paper summary
- `predictions.md` — written **before** running
- `src/` — code
- `results/` — plots, logs, findings
