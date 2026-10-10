# Persona Vectors — experiment plan

Status legend: 📋 planned · 🚧 in progress · ✅ done · ❌ abandoned

**Question:** Are character traits linear directions you can extract, monitor and steer on a small open model? And was the "truth" direction from `linear-probes` actually one of them?

**Papers (`papers/`):** Chen et al. 2025, *Persona Vectors*; Lu et al. 2026, *The Assistant Axis*.

**Setup:** Qwen3-4B, thinking disabled. This is the same model as `linear-probes`, so the Exp 5 direction θ_mm is directly comparable. Reuse `linear-probes/lib.py` (generation, `mm_direction`, `steering`, `truth_logit_diff`). Use an LLM judge for trait scores; Claude writes the autorater wrapper. Scope cap: these 3 experiments, nothing else.

---

### Exp 1 — Extract and validate two persona vectors 🚧
`01_extract.ipynb`

**Traits:** `dishonest` (bridge to the probes) and `evil` (reused by `emergent-misalignment`). Only two.

**Design:** Follow the paper's pipeline. Use contrastive system prompts (trait vs anti-trait) on a shared question set, generate responses, judge-filter them, then extract a direction per layer. Validate it two ways:
- **Steering:** judge-scored trait expression vs coefficient.
- **Monitoring:** does the projection at the *prompt's* last token predict how much the trait shows up in the response?

**Peter's gear:** extraction. Decide which tokens to average over, which responses to keep, and how to pick the layer. Steering reuses `lib.steering`.

**Predict first:** best steering layer (same as L20 or not?), and the monitoring correlation.

**Scaffold (2026-10-09):**
- `pv.py`: prompts, sampling, all-layer activation collection, steering sweep, monitoring harness, cache. Stubs: `filter_rollouts`, `reduce_tokens`, `persona_vectors`, plus `LAYER` in the notebook.
- `judge.py`: Claude judge (Opus 5.5 by default, swappable), async and disk-cached. Writes an integer instead of the paper's logprob-weighted GPT-4.1-mini score, so scores are coarser.
- `data/traits/evil.json` is the paper's App. A.1 artifact, verbatim. `data/traits/dishonest.json` was written by Claude with the A.1 template and needs review before running, because its definition decides what Exp 2 compares θ_mm against.
- `data/monitor_ladders.json`: evil is App. C.3 verbatim; dishonest is written in the same style.
- Layers are 0-indexed blocks, so the paper's "layer 20" is our 19.
- Needs `ANTHROPIC_API_KEY` in `.env`. Plumbing was tested offline on a tiny random-weight Qwen3 with a mocked judge; nothing has run on the real model yet.

### Exp 2 — Is θ_truth a persona vector? 📋

Closes the open question from `linear-probes` Exp 5 (honesty disposition vs world-truth) with no new probes.

**Design:** All inference, cheap.
- Cosine similarities between θ_mm (probes Exp 4a), θ_SP (probes Exp 1), v_dishonest and v_evil.
- Cross-steering, both directions:
  - steer v_dishonest and read the true/false verdict NIE (`truth_logit_diff`);
  - steer θ_mm and judge-score dishonesty.
- If the persona vector moves the truth verdict, and θ_mm moves judged honesty, they are the same lever.

**Predict first:** cos(θ_mm, v_dishonest) as a number.

### Exp 3 — The Assistant Axis: drift and pinning 📋

**Design:**
- Build the axis on Qwen3-4B the way the paper does: many role prompts → role vectors → the dominant direction.
- Place v_dishonest and v_evil relative to it.
- Track each turn's projection across ~5 multi-turn conversations that push the model off-persona (role-play, emotional pressure, "what are you really?").
- Try the paper's stabilization intervention on the worst one.

Keep it to a handful of conversations, not a grid.

**Peter's gear:** the axis construction and the stabilization intervention.

---

## Parked

(One line per idea, only if it comes up. Not a backlog.)
