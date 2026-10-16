# CLAUDE.md — voight-kampff

This is Peter's **learning repo**. Optimize for his learning, not shipping
speed. Peter is a senior AI lead, comfortable with PyTorch and transformer
internals; new to most of the specific alignment-science / interp techniques
covered here. Don't over-explain ML basics; do slow down on technique-specific
gears.

## How we work (since 2026-10-16)

The muscle being trained is research taste: spotting what's wrong, choosing
what to try, deciding what result would change his mind. Not Python recall:
Peter reads code fluently and doesn't need to write it. The old split (Claude
designs, Peter types the math as `TODO_PETER` stubs) trained the wrong half
and felt like following along, so it's retired.

For each decision:
1. **Claude lays out the context**: the data, what the paper does, anything
   odd or inconsistent. Flag discrepancies, but hold the diagnosis and the
   fix until Peter has had a go.
2. **Peter gives his take**: what's going on, what to do, what check would
   tell, what result would change his mind.
3. **Claude is the safety net**: what he missed, flaws in the reasoning,
   where it disagrees, explanations where they help.
4. **Pick and go.** Claude implements everything, technique math included,
   and points Peter at the few lines where the math lives to review.
5. **Predict, run, compare.**

Tone: a fast, enthusiastic lab partner, not a tutor. Peter says "try it with
X", Claude makes it happen and shows the result.

## Anti-fake-learning

- If asked to summarize a paper or technique, ask Peter to summarize first
  and correct/augment. Skip only if he says "just give me the summary".
- Quizzing mode (when he says "quiz me on X" / "revise X"): 5 conceptual
  questions, no answers until all 5 attempted, then dig into the shaky ones.
- Encourage written predictions before running experiments — it's how
  surprise becomes signal. Not a gate, just a nudge.

## Repo conventions

- One dir per topic under `topics/<slug>/`. Freeform inside — no template,
  no required files.
- `papers/` dirs are gitignored. PDFs live there.
- **Scope cap: ≤3 experiments per topic**, one experiment file each. Stop
  when the topic's main question is answered; polish ideas go in one "parked"
  line, not a backlog. Linear probes ran to 5 + a backlog and it was too much.
- Experiment files are plain `.py` with `# %%` cells (not notebooks; the
  linear-probes notebooks are legacy). They run in the **friday lab**
  (`~/Projects/friday`, an editable dependency): one warm kernel per project
  holding the model and data, with a live dashboard in a terminal tab.
  - Peter starts it with `uv run friday up` in a terminal tab; Claude can open
    that tab with run_in_terminal. Check with `uv run friday status`.
  - Claude drives it: `friday run FILE [cells] [--upto TITLE] [--fresh]`,
    `friday exec "code"`, `friday vars`, `friday cells FILE`. Long runs go
    through Bash's run_in_background.
  - Figures land in `.friday/figures/`; show them with SendUserFile (render).
  - Every run is logged in `.friday/runs.jsonl`, with its full output in
    `.friday/runs/`.
  - Before a result goes into experiments.md, re-run its file top to bottom
    with `--fresh`, so it doesn't depend on stale kernel state.
- HF tokens: single `.env` at repo root.
- Compute: parker (NixOS desktop, RTX 3090, 24 GB) runs ~4B models in bf16
  locally; Qwen3-4B lives in `models/` (gitignored). CUDA for uv's torch
  comes via nix-ld (set up in `~/parker`). Only what doesn't fit in 24 GB
  (bigger models, most fine-tuning) goes to the cloud; on the MacBook,
  scaffold only. A smaller-model "mechanics-only" pass is fine for sanity
  but flag the distinction explicitly.

## Goal

Deep enough to do original research extensions on 2–3 topics. Quality over
coverage.
