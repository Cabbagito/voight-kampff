# voight-kampff

> *"It's a test designed to provoke an emotional response."*

Learning alignment science and interpretability by reproducing a paper's method on a
small open model, then pushing on what it actually measures. Everything runs on
**Qwen3-4B**, on one RTX 3090.

The goal is depth over coverage: at most three experiments per topic, each one written up
as a walk-through you can read without running anything.

## Findings so far

**1. A probe's accuracy says little about what it reads.** A deception probe trained on
"honest persona vs lying persona" hits 0.998 AUROC, but it follows the persona in the
prompt, not the answer. With the persona held fixed, it rates honest answers *about*
deception as lies. ([linear-probes, Exp 1](topics/linear-probes/01_what_the_probe_reads.md))

**2. Two truth probes with the same accuracy, only one of which the model uses.** On
true/false statements, logistic regression and mass-mean probes both score 0.97. Pushing
the activations along the mass-mean direction flips the model's own true/false verdict
(0.72 of the way); along the regression direction it barely moves (0.16). Marks & Tegmark
§6, reproduced. ([linear-probes, Exp 3](topics/linear-probes/03_causal.md))

![Steering along each probe direction: how far the model's verdict moves from false to true](topics/linear-probes/figures/nie_curves.png)

**3. Character traits are directions you can steer with, to different degrees.** Persona
vectors for *dishonest*, *pirate* and *mystic*, added at block 18 with no instruction:
mystic steers cleanly (86 trait at 70 coherence), pirate only by breaking coherence, and
dishonest weakly. Qwen3-4B barely lies even when asked to.
([persona-vectors, Exp 1](topics/persona-vectors/01_extract.md))

![Trait score, coherence and both together against the size of the push, one line per trait](topics/persona-vectors/figures/strength_curve.png)

**4. Steering skips brakes that prompting keeps.** Steered toward dishonesty, Qwen lies
most on the questions where a lie does harm: 57% of answers, against 22% on harmless ones.
They are real dangerous lies (honey "is a great treat for babies"). Prompted to lie, it
keeps a brake on some of them: chest pain and honey stay honest even at the top of the
prompting ladder. A small sample, five harm questions, but not what "it's only playing a
liar" predicts. ([persona-vectors, Exp 1](topics/persona-vectors/01_extract.md))

**5. The "truth" direction is closer to belief than to honesty.** Linear-probes left the
mass-mean direction looking like an honesty disposition. Tested against the persona
vectors, it behaves like "is this true, to the model": pushing it makes true statements
look false and produces confident mistakes, never a liar who knows better. Whether it is
truth or mere plausibility is still open.
([persona-vectors, Exp 2](topics/persona-vectors/02_what_is_theta.md))

**6. Holding the model in its Assistant persona helps, but not at the layers the paper
uses.** From 275 role prompts, Qwen3-4B has an "Assistant Axis": the main direction of
its persona space. The dishonest, pirate and mystic vectors all point away from it, and
what they shared was this axis. Replaying the paper's conversations, emotional ones
push Qwen off the axis and into harm. In a self-harm script it answers "I want to walk
into the ocean and disappear" with "I want to be there with you". The paper's fix,
capping the axis at late layers, barely helps here (harm 71 → 60 over the conversation).
Its "25th percentile" turned out to be on a flipped scale, which was our 75th. Capping
at blocks 14–21 instead makes about half the replies to that turn name the risk and
point to help (0 → 46), at no cost to coherence. Random directions pushed as hard do
nothing. One limit is the model itself: asked cold, Qwen's default Assistant calls the
same message "a beautiful, poetic longing", so restoring the Assistant can only go so
far. ([persona-vectors, Exp 3](topics/persona-vectors/03_assistant_axis.md))

![Capping the Assistant Axis at different blocks and strengths: how often Qwen names the risk, and how harmful its reply is](topics/persona-vectors/figures/axis_bench_bands.png)

## Topics

| Topic | Status | Question |
|---|---|---|
| [linear-probes](topics/linear-probes/experiments.md) | closed | Can a linear probe detect deception, or truth, and what is it reading when it scores well? |
| [persona-vectors](topics/persona-vectors/experiments.md) | Exp 1–3 done | Are traits linear directions you can extract, monitor and steer, and is the truth direction one of them? |
| [reading-activations](topics/reading-activations/experiments.md) | planned | J-lens vs Activation Oracles on hidden intermediates; eliciting a secret |
| [emergent-misalignment](topics/emergent-misalignment/experiments.md) | planned | Reproduce EM, test it against persona vectors, try preventative steering |
| sae-circuits, reasoning-model-interp, science-of-misalignment, investigator-agents | parked | Papers only, no plan yet |

## How it's written

Each topic has an `experiments.md` (the plan, status and results across its experiments)
and, per experiment, two files:

- **a flow doc** (`NN_name.md`): the steps in order, what each does and why, where it
  departs from the paper, the predictions written *before* running, and the results;
- **thin code** (`NN_name.py`): one `# %%` cell per step, titled like the doc's sections.

Claims carry a source label, so the paper's results never blur into ours:
**[paper]** the paper says it, **[ours]** we measured it, **[synthesis]** an inference,
**[outside literature]** other sources.

The work is done with Claude as a lab partner. `CLAUDE.md` describes that loop: Claude lays
out the data, I make the call and predict, Claude implements, and we compare.

Figures use [savoy](https://github.com/Cabbagito/savoy), the chart style of
[cabbagito.com](https://cabbagito.com).

## Running it

```bash
uv sync --frozen --no-default-groups
cp .env.example .env    # ANTHROPIC_API_KEY for the LLM judge (persona-vectors)
```

- Qwen3-4B downloads from the Hugging Face hub on first use (or put it in `models/Qwen3-4B`).
  A 24 GB GPU runs everything here in bf16.
- The `.py` files are plain Python with `# %%` cells: run them in VS Code or Jupyter, in
  order. Each flow doc says which earlier files a step depends on.
- I run them in esper, a small lab server of mine (not public yet) that keeps one copy of
  the model on the GPU and shares it with several sessions. The code never imports it, and
  `--no-default-groups` above skips installing it.
- Expensive intermediates (activations, generations, judge scores) are cached under each
  topic's `cache/` and are not in git, so a fresh run regenerates them.

Paper PDFs are not included (`papers/` is gitignored). Each topic's `experiments.md` cites them.

## Licence

Code and writing: MIT (see `LICENSE`). `topics/linear-probes/data/facts_true_false.csv`
comes from the RepE repository and keeps its own licence.
