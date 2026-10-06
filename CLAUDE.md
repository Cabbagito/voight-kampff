# CLAUDE.md — voight-kampff

This is Peter's **learning repo**. Optimize for his learning, not shipping
speed. Peter is a senior AI lead, comfortable with PyTorch and transformer
internals; new to most of the specific alignment-science / interp techniques
covered here. Don't over-explain ML basics; do slow down on technique-specific
gears.

## Implementation split

- **Implement fully** (you): glue, configs, data loading, plotting,
  scaffolding, autorater wrappers, eval harnesses, dataset prep, viz.
- **Leave as stubs for Peter**: the technique-defining substance — probe
  training loops, hook functions, attribution gradients, steering math,
  persona extraction, anything where the *gears* live. Mark with
  `# TODO_PETER:` and a 1–2 line shape hint (tensor shapes, candidate library
  fns). **Do not write the answer.**
- If unsure which side, ask.
- If Peter says "just implement it" on a substance stub: push back once. If
  he confirms, do it but flag what gears he's skipping.

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
- **Scope cap: ≤3 experiments per topic**, one notebook each. Stop when the
  topic's main question is answered; polish ideas go in one "parked" line,
  not a backlog. Linear probes ran to 5 + a backlog and it was too much.
- HF tokens: single `.env` at repo root.
- Most sections need A100-class GPU; Peter's on a Mac. Default: scaffold
  locally, run on cloud. A smaller-model "mechanics-only" pass is fine for
  sanity but flag the distinction explicitly.

## Goal

Deep enough to do original research extensions on 2–3 topics. Quality over
coverage.
