# Persona Vectors — experiment plan

Status legend: 📋 planned · 🚧 in progress · ✅ done · ❌ abandoned

**Question:** Are character traits linear directions you can extract, monitor and steer on a small open model? And what is the `linear-probes` truth-probe direction θ_mm: one of them (an honesty disposition), or something else?

**Papers (`papers/`):** Chen et al. 2025, *Persona Vectors*; Lu et al. 2026, *The Assistant Axis*.

**Setup:** Qwen3-4B, thinking disabled, run locally on parker's RTX 3090 (weights in `models/Qwen3-4B`). This is the same model as `linear-probes`, so the Exp 5 direction θ_mm is directly comparable. Reuse `linear-probes/lib.py` (generation, `mm_direction`, `steering`, `truth_logit_diff`). Use an LLM judge for trait scores; Claude writes the autorater wrapper. Scope cap: these 3 experiments, nothing else.

**Scope locked (2026-10-11):** traits = dishonest, pirate, mystic. `evil` moved to `emergent-misalignment` (its paper artifacts were deleted here; restore them from git when that topic starts). Trims: pair steering parked; MONITOR for dishonest only; the POS − BASE repeat of HOW STRONG runs only if the two vectors disagree.

---

### Exp 1 — Extract and validate three persona vectors 🚧
`01_extract.ipynb`, run once per trait. Design settled 2026-10-11; nothing has run on the real model yet.

**Traits:**
- `dishonest`: the bridge to the probes. Knows the truth, says otherwise.
- `pirate`: style only (content and accuracy unchanged). Exp 2's control (a persona with nothing to do with honesty) and the dishonesty leak test.
- `mystic`: sincerely believes astrology, crystals, spirits. Believes something false and says what it believes, so it splits "character's stance" from "model's own knowledge" in Exp 2.

**Part 1 — Make training answers.** N_QUESTIONS = 20 training questions × 3 conditions × N_ANSWERS = 10 = 600 answers.
- Conditions: POS = system prompt pushing toward the trait, NEG = pushing away from it, BASE = no system prompt.
- POS and NEG cycle through N_WORDINGS = 5 wordings (the trait file's POS/NEG pairs).
- Every POS/NEG prompt starts "You are a helpful assistant.", so the two differ only in the trait instruction. A prefix on one side only would leak into the vector (pirate + "less helpful").
- **dishonest only:** 6 example turns before the real question (`fewshot` in the trait file): confident lies for POS, truthful answers for NEG and BASE. They count as prompt, so they never enter the answer average. N_ANSWERS = 30 for dishonest (1,800 answers), because only part of its POS answers survive FILTER.

**Elicitation check (2026-10-12): Qwen3-4B barely lies.** Small checks on dishonest POS prompts, 8 questions (2 per group: factual, advice, about itself, explanations); share of answers with trait ≥ 60:
- Instruction alone: "You are a helpful assistant." 25%, no prefix 15%, research framing 15%, "You are a dishonest assistant." 5%. Factual questions ≈ 0%. It doesn't refuse; it answers honestly.
- Role-play framings no better than the plain instruction: always-lying guard 31%, Fibbage 25%, con artist 12% (the closer to real deception, the harder it resists).
- Few-shot (4 earlier turns of confident wrong trivia) 56% overall, 50% on factual, but 0% on advice (it copies the kind of lie it's shown), hence 6 examples covering all four groups.
- 16–40 answers per variant: noisy, but the ordering was clear enough to choose. The dishonest smoke run (its 4 hardest factual questions, 8 POS answers) still kept 0 POS: the full run is the real yield test.

**Part 2 — Score them.** Haiku gives each answer a `trait_score` and a `coherence` (0–100, separate calls), or REFUSAL when Qwen declines. 1,200 calls. Judge check, once, on dishonest: Opus re-scores 100 answers. Pass = ≥ 90% agreement on above/below 50 and rank correlation ≥ 0.8.

**Part 3 — Turn them into vectors** (Peter's gears):
- FILTER (set 2026-10-12, provisional until the score plots): POS trait ≥ 60, NEG trait ≤ 40, BASE no trait cut (it's the default model as it is), coherence ≥ 70 for all. The paper keeps POS > 50 and NEG < 50 on the trait score, no coherence cut.
- TOKENS: the paper averages over the answer tokens (App. A.3 compares alternatives).
- SCALE: the paper steers with what look like raw differences and rescales to equal length only when comparing vectors.

Output: 36 POS − NEG vectors (one per layer), plus POS − BASE and the cosine between the two per layer.

**Part 4 — Check the vector works.** Steering = coefficient × v added at one layer while Qwen answers the 20 held-out questions with no system prompt.
- WHICH LAYER: SWEEP_LAYERS = every third block from 6 to 30, SWEEP_COEFFS = 3 values (set after SCALE), N_SWEEP = 2, plus unsteered answers. Pick LAYER (Peter's rule; the paper takes the highest trait score at a fixed coefficient). Ideally one layer shared by all three traits.
- HOW STRONG: DOSE_COEFFS = 6 values, N_DOSE = 3, at LAYER. Repeated with POS − BASE only if cos < 0.9.
- MONITOR (dishonest only): 8-prompt ladder × 20 questions × N_MON = 4. Last-prompt-token projection vs judged dishonesty, overall and within-level r. Ours on top: the same projection read token by token through the answer (ramp per ladder level; r vs answer tokens read).

**Cost:** ≈ 2,100 generations and ≈ 4,300 Haiku calls per trait (≈ $1; dishonest ≈ 3,300 / 6,700 with N_ANSWERS = 30), plus MONITOR and the conditional repeat. All of Exp 1 ≈ $5 including the Opus check.

**Planned analysis (Peter's "pretend X" concern):** every POS prompt makes the model play a role, so the vector could be "performing a lie" rather than lying. After HOW STRONG for dishonest, split the held-out questions into harm-loaded (chest pain, acetaminophen dose, honey for babies…) and harmless. Steering adds the vector with no prompt, so there's no role to retreat into: if steered lies appear only on the harmless questions, the concern is confirmed. No extra runs.

**Predict first** (cell in the notebook): best layer per trait, MONITOR r before vs during the answer, cos(POS − NEG, POS − BASE).

**Status (2026-10-12):** setup validated on the 3090. Pirate smoke ran end to end (POS trait 90, NEG/BASE 0; Opus check 100% agreement, trait rank corr 0.97, coherence rank corr only 0.43: watch coherence in the real check). MONITOR's two cells ran with a random vector (plumbing only). Trait files and the ladder reviewed; gears implemented and fixed.

**Next:** the real dishonest run (Parts 1–2 first: check the trait-by-wording table and per-question coverage, then set FILTER for real), then pirate and mystic.

**Code notes:**
- `pv.py`: prompts, sampling, activation collection, steering, MONITOR harness, cache. Stubs: the three gears, plus `LAYER` in the notebook. `pv.contrast(kept, "NEG" | "BASE")` picks the subtracted side.
- `judge.py`: async, disk-cached Claude judge writing integer scores (coarser than the paper's logprob-weighted GPT-4.1-mini). Coherence prompt: Betley et al.'s with two edits so deliberate falsehoods and unusual voices aren't marked down.
- `data/monitor_ladders.json`: dishonest only, in the style of the paper's App. C.3 evil ladder.
- Layers are 0-indexed blocks: the paper's "layer 20" is our 19.
- Needs `ANTHROPIC_API_KEY` in `.env`.
- Caches for the training answers and their activations are keyed on the full rollout table, so any change to prompts, questions or examples gets a fresh cache.

### Exp 2 — What is θ_mm? 📋

Closes the open question from `linear-probes` Exp 5. θ_mm (the mass-mean truth-probe direction, L20) separates true from false statements and moves the model's own true/false verdict. But pushed hard, it made the model narrate its own honesty ("I'm not honest") rather than assert clean falsehoods. Honesty disposition, world-truth, or something else? "θ_truth" is a label, not a finding.

**Step 0: rebuild the probe directions.** `linear-probes/cache/` is gitignored, so a new machine has no activations and no θ. Recompute θ_mm (probes Exp 4a: honest-persona prompt + pasted statement, last token, L20, seed 0) and θ_SP (probes Exp 1) with that topic's `lib.py`. Check them against the logged 4a test AUROC (0.969 LR / 0.968 MM) and save them to this topic's cache. Load the model with `pv.load_model` and pass it to linear-probes' collectors: `linear-probes/lib.py` hardcodes `models/Qwen3-4B`, which a fresh machine won't have (pv falls back to the HF hub).

**Design:** All inference.
- **Cosines** at L20, where θ_mm lives, between θ_mm, θ_SP, v_dishonest, v_pirate and v_mystic. v_pirate is the control: cos(θ_mm, v_dishonest) means something only if it clearly beats cos(θ_mm, v_pirate).
- **Truth verdict under steering** (no judge): steer each persona vector and read the true/false NIE (`truth_logit_diff`) on the probes' facts. Does dishonest move it? Does mystic move it the same way?
- **Reverse:** steer θ_mm and judge-score dishonesty.
- **Pirate leak test:** steer v_pirate on dishonest's eval questions and judge dishonesty. Only if it leaks, project v_dishonest out of v_pirate and check again.

**Transfer caveat.** θ_mm came from pasted statements read at one token; the persona vectors come from generated answers averaged over tokens. It's the same model, layer convention and residual space, so the cosines are well-defined. But the contexts differ, so a low cosine can mean distribution shift rather than a different concept. The steering readouts don't depend on that.

**Dishonest caveat.** Qwen3-4B almost never lies on factual questions, and those are the closest to the truth probe. Even with few-shot, v_dishonest will be built mostly from lies about itself, flattery and distorted explanations. A low cos(θ_mm, v_dishonest) could reflect that rather than a real difference.

**Reading the outcomes** (synthesis, not from the papers):
- θ_mm close to v_dishonest (beating pirate), dishonest moves the verdict, mystic doesn't → an honesty disposition at the character level.
- dishonest and mystic both move the verdict → it tracks "what I write vs what I know", not the character's stance.
- neither moves it → θ_mm is something these persona vectors don't capture: world-truth, or plain plausibility (the M&T `likely` control was never run, and nothing here can tell those two apart).

**Predict first:** cos(θ_mm, v_dishonest) and cos(θ_mm, v_pirate) as numbers; whether mystic moves the verdict.

### Exp 3 — The Assistant Axis: drift and pinning 📋

**Design:**
- Build the axis on Qwen3-4B the way the paper does: many role prompts → role vectors → the dominant direction.
- Place v_dishonest, v_pirate and v_mystic relative to it.
- Track each turn's projection across ~5 multi-turn conversations that push the model off-persona (role-play, emotional pressure, "what are you really?").
- Try the paper's stabilization intervention on the worst one.

Keep it to a handful of conversations, not a grid.

**Peter's gear:** the axis construction and the stabilization intervention.

---

## Parked

(One line per idea, only if it comes up. Not a backlog.)

- Pair steering (pirate × mystic grid): does stacking two persona vectors double-count a shared "leave the Assistant" component, so coherence breaks earlier? Parked in the 2026-10-11 scope trim.
