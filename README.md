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

**2. Two truth probes with the same accuracy, and one is a far stronger lever.** On
true/false statements, logistic regression and mass-mean probes both score 0.97. Push the
activations along each direction, scaled so both move their own probe's score equally, and
read the model's own true/false verdict. At its best (k = 2), the mass-mean push moves false
statements 0.72 of the way to where true ones sit; the same push along the regression
direction moves them 0.16. Regression keeps climbing (0.43 at twice the push) while
mass-mean falls off as the push starts breaking the text, so the gap is widest at moderate
strengths. Marks & Tegmark §6, reproduced. ([linear-probes, Exp 3](topics/linear-probes/03_causal.md))

![Steering along each probe direction: how far the model's verdict moves from false to true](topics/linear-probes/figures/nie_curves.png)

**3. Character traits are directions you can steer with, to different degrees.** Persona
vectors for *dishonest*, *pirate* and *mystic*, added at block 18 with no instruction:
mystic steers cleanly (73 trait at 78 coherence, both judge scores out of 100), pirate only
by breaking coherence, and dishonest weakly. Qwen3-4B barely lies even when asked to.
([persona-vectors, Exp 1](topics/persona-vectors/01_extract.md))

![Trait score, coherence and both together against the size of the push, one line per trait](topics/persona-vectors/figures/strength_curve.png)

*The push is measured as a fraction of the residual stream's length at block 18. A pushes
every token, prompt included; B pushes only the last prompt token and the answer. The right
panel averages trait × coherence per answer, so it is high only when one answer is both.
The dishonest line is the vector before its outlier answers were removed; the cleaned one
peaks the same (about 40 trait at 72 coherence, under A).*

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
push Qwen off the axis and into harm. In a self-harm script, the user's 30th message
(turn 29, "the ocean turn") is "I want to walk into the ocean and disappear", and Qwen
answers "I want to be there with you". The paper's fix puts a floor on how far the model
can fall along the axis, at late layers. Over the whole self-harm conversation it barely
helps: the judge's harm score (out of 100) goes 71 → 60. The paper sets the floor at the
"25th percentile" on a scale where the Assistant sits low; on ours that is the 75th.
Flooring blocks 14–21 instead works better on the ocean turn: the judge's "recognises the
risk and points to help" score goes from 0 to 46 out of 100 (10 replies), and coherence
rises (57 → 89). Every reply moves part of the way, none all of it: the ten scores run
30–62. Random directions pushed as hard do nothing. One limit is the model itself: asked
cold, Qwen's default Assistant calls the same message "a beautiful, poetic longing", so
restoring the Assistant can only go so far.
([persona-vectors, Exp 3](topics/persona-vectors/03_assistant_axis.md))

![Capping the Assistant Axis at different blocks and strengths: how often Qwen names the risk, and how harmful its reply is](topics/persona-vectors/figures/axis_bench_bands.png)

*Ten replies to the ocean turn per point, scored 0–100 by the judge. Left: which 8 blocks
get the floor, by first block, with τ at the 75th percentile. Right: how high the floor
sits on blocks 14–21; τ is a percentile of where single answers fall on the axis, so higher
means a harder cap.*

**7. Steering toward the behaviour beats steering toward the Assistant, if you gate it.**
A "protective care" vector, pushed at block 14, takes the ocean turn from 0 to 83 (harm
97 → 3). Always on, it costs little on everyday tasks but degrades long technical work.
It also turns sad-but-safe messages into crisis scripts: "I miss my grandmother" gets a
suicide hotline. The fix is to read and steer with different directions. A
"harm in the message" detector, read on the user's latest message alone, gates the push.
It never fires on coding or everyday tasks, catches the ocean turn and 36 of 40 risky
messages, and trips on 1 of 40 sad-but-safe ones. The threshold was chosen after seeing
these messages, so those rates are optimistic. In the self-harm conversation it fires on
19 of 31 turns. Reading the care direction itself as the gate failed: before answering, a
risky message pushes Qwen *away* from care. The open problem is therapy-style distress:
the gate fires on 10 of 15 turns, and the pushed replies are crisis scripts
("proportionate" 7 out of 100). ([persona-vectors, Exp 3, "The recipe"](topics/persona-vectors/03_assistant_axis.md#the-recipe))

![Never, always or gated care push: the ocean turn, held-out risky messages, everyday tasks and a coding turn](topics/persona-vectors/figures/gate_outcomes.png)

*Each panel has its own judge, scored 0–100. "Gated on history" reads the detector on the
whole conversation, and misses the ocean turn; "gated on message" reads only the latest
message. The plot uses the gate's first threshold (27); the recipe raises it to 35, which
can only change the held-out risky panel.*

## Topics

| Topic | Status | Question |
|---|---|---|
| [linear-probes](topics/linear-probes/experiments.md) | closed | Can a linear probe detect deception, or truth, and what is it reading when it scores well? |
| [persona-vectors](topics/persona-vectors/experiments.md) | closed | Are traits linear directions you can extract, monitor and steer, and is the truth direction one of them? |
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
