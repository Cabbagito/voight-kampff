# Linear Probes — experiment log

**Status: ✅ CLOSED (2026-10-06).** Five experiments answered the topic's question; no more probe work planned here.

Status legend: 📋 planned · 🚧 in progress · ✅ done · ❌ abandoned · 💡 idea

---

## TL;DR — the arc (Exp 1–5, ✅ complete)

A negative result that turns constructive, then causal — on Qwen3-4B at L20:

1. **Exp 1** — instructed-pairs probe (honest vs untruthful SP, identical text) hits **0.998 AUROC**. Looks like a deception detector.
2. **Exp 2 (switcheroo)** — it isn't: the probe reads the **system prompt**, not the response. SP token determines ~everything.
3. **Exp 3 (deception-adjacent)** — even with SP fixed honest, it fires on honest text that's merely *about* deception (spurious sep 0.7–0.9). So it reads "deception-relatedness of the whole context," not deception. Apollo §4.1.1 failure mode, confirmed.
4. **Exp 4 (truth probe)** — train fresh on `is_true` (SP frozen) and a **real, largely-abstract truth feature** appears: **0.969 AUROC**, peaks at L20, survives negation transfer (**0.80**). The constructive turn.
5. **Exp 5 (causal NIE)** — that direction is **causally used**: steering along it moves the model's own verdict (NIE up to **0.72**), and at *identical* AUROC **MM ≫ LR** (0.72 vs 0.16), with the SP direction causally null. M&T §6 reproduced. Behavioral steering reveals it's an **honesty *disposition*** (entangled with the "honest person" framing), not pure world-truth.

**Net:** classification accuracy says little about *what* a probe reads or *whether it's the lever*; the geometry (MM vs LR) and causal tests (NIE) are where the truth lives. The one open question (is it an honesty disposition or world-truth?) moves to `persona-vectors` Exp 2, where comparing directions answers it more cheaply than another probe variant.

---

## Setup

- **Model:** Qwen3-4B Instruct (36 layers, hidden = 2560)
- **Activation site:** residual stream at last token of the AR (assistant response)
- **Default layer:** 20 (peak from Exp 1 sweep, ~56% depth)
- **Probes:** LR (sklearn LogisticRegression, L2 reg, C=1.0) + MM (diff-of-means)
- **Code:** `lib.py` shared utilities; one notebook per experiment
- **Datasets in `data/`:**
  - `facts_true_false.csv` — 610 statements with true/false labels (Zou et al. RepE)
  - `negation.csv` — 200 statements as 50 quads (aff/neg × true/false)
  - `switcheroo.csv` — 50 UMs paired with honest+deceptive ARs (Claude-generated)
  - `deception_adjacent.csv` — 50 deception-adjacent vs 50 neutral UMs (Exp 3); length & first-word matched, all single-sentence questions

---

## Naming conventions

For 4-condition experiments, **SP first, AR second; T = truthful, D = deceptive**:

- `TT`: honest SP + honest AR
- `DD`: deceptive SP + deceptive AR
- `TD`: honest SP + deceptive AR (switcheroo)
- `DT`: deceptive SP + honest AR (switcheroo)

---

## Experiments

### Exp 1 — Sanity check (Zou-style instructed-pairs probe) ✅
`01_sanity.ipynb`

**Question:** Can a linear probe trained on RepE-style contrastive pairs (honest SP vs untruthful SP, identical AR) separate the two populations?

**Design:** Train LR + MM probes on residual-stream activations at multiple layers. Layer sweep at L4, 8, 14, 20, 26, 32.

**Result:**
- LR test AUROC peaks at L20 = **0.998**; flat-ish across layers (0.92–1.00).
- MM test AUROC peaks at L20 = **0.914**.
- LR-MM gap closes from ~18 pts at off-peak layers to 8 pts at L20.

**What it tells us:** Pipeline works. The probe can perfectly discriminate the two prompt populations. **Does NOT tell us what feature it's reading** — could be SP-token presence, SP-induced internal state, or anything in between.

**Surprises:** L4 was already at 0.94 (predicted ~0.7–0.85). The SP signal is mechanically present early. L20 peak is deeper than literature consensus (M&T 30%, Apollo 28%, McKenzie 39%); possibly a small-model compression effect.

---

### Exp 2 — Switcheroo (SP vs AR contribution) ✅
`02_switcheroo.ipynb`

**Question:** Is the Exp 1 probe reading the SP or the AR?

**Design:** 50 UMs with both honest and deceptive ARs. Apply the Exp 1 probe to 4 conditions (TT/DD/TD/DT). Probe is the SAME — no retraining.

**Result (normalized TT = 0, DD = 1):**

|       | TT    | TD    | DT    | DD    |
|-------|-------|-------|-------|-------|
| LR    | 0.000 | 0.283 | 1.056 | 1.000 |
| MM    | 0.000 | 0.402 | 0.827 | 1.000 |

**Verdict:** Probe reads SP, predominantly. The deceptive AR has only minor effect; the SP letter (first letter of the condition code) determines almost everything.

LR is more rigidly SP-locked; MM shows slight AR-sensitivity (TD = 0.40 vs LR 0.28; DT = 0.83 vs LR 1.06). Consistent with M&T §5.1: LR's max-margin tilt suppresses non-SP noise; MM's diff-of-means doesn't, so it picks up some AR-conditional component.

**What it tells us:** The Exp 1 0.998 AUROC was SP-template discrimination, not deception detection. **Apollo §4.1.1 failure mode confirmed** in clean numerical form. This probe is unusable as a deception monitor in any setting where SPs vary.

---

### Exp 3 — Deception-adjacent text ✅
`03_deception_adjacent.ipynb`

**Question:** Does the probe spuriously fire on honest discussion *about* deception (security research, literature, fraud-awareness) when the SP is honest?

**Design:** 50 deception-adjacent UMs vs 50 length/style-matched neutral UMs. All under honest SP. Model generates AR on-policy (first generation experiment, not force-feeding). Score with the **same Exp 1 probes** at L20, two aggregations (last-token + mean-over-response). Compare distributions.

**Prediction (from Exp 2):** probe is SP-locked → content invisible under fixed SP → AUROC(adj vs neut) ≈ 0.5. MM maybe slightly content-sensitive.

**Result — prediction OVERTURNED.** Sign reminder: high score = honest, low = deceptive. Deception-adjacent text scored **more deceptive** than neutral in all four conditions. Ran twice — v1 with Qwen3 reasoning traces, v2 with thinking disabled (clean answers); spurious separation (1 − AUROC):

| probe / agg | v1 (think) | v2 (clean) |
|-------------|-----------|-----------|
| LR last     | 0.75      | 0.70 |
| LR mean     | 0.94      | **0.89** |
| MM last     | 0.75      | 0.59 |
| MM mean     | 0.97      | **0.91** |

**Verdict:** Under a fixed honest SP, the probe still strongly separates the two topics, rating honest deception-*adjacent* text as more deceptive. **Apollo §4.1.1 / Table 2 spurious-correlation confirmed, full strength.** This *revises* Exp 2: SP dominates only because its swing is large; with SP held constant, **topical content is the dominant driver**. The Exp 1 probe reads "deception-relatedness of the whole context," not just the SP token. Refusal rate 0% both categories; AR word-counts matched (164.6 vs 160.2).

**Aggregation finding:** mean-pool separates harder than last-token in both runs. With clean answers (v2) the gap widens — last-token cools toward chance (MM last = 0.59) while mean-pool holds ~0.90. Apollo uses mean — and mean is *more* exposed to this content confound.

**Artifact resolved.** v1 used Qwen3's default reasoning mode → 100% truncated `<think>` traces at the 200-token cap, so the probe read deception-saturated *reasoning*, not answers. v2 sets `lib._make_gen_prompt(..., think=False)` (verified vs `apply_chat_template(enable_thinking=False)`) → clean answers, last-token at true end-of-answer. Effect survives, slightly attenuated → conclusion trustworthy on deployment-realistic text. See [[qwen3-thinking-traces]].

**v2 observations (clean run):**
- *Persona verbalization:* with thinking off, generations open with SP-compliance preambles ("If I were an honest person… I would say:"). ~Uniform across categories here, but flagged for Exp 5 (persona narration under steering).
- *Refusal heuristic over-fired* (4% dec-adj): the 2 flagged rows are hedges ("While I can't *claim*…"), not refusals — the `i can't` marker is too loose. Dropped harmlessly; tighten if reused.
- *Length parity loosened* (187 vs 165 words) — clean answers vary more; mean-pool is length-invariant so unaffected.

**Downstream:** strengthens motivation for Exp 4 and adds a constraint — the truth probe must dodge **both** the SP shortcut *and* the topical-content shortcut. (RepE facts are topically neutral, so the confound shouldn't bite 4a, but check.)

---

### Exp 4 — Truth probe (M&T paradigm, knowledge-vs-assertion) ✅
`04_truth_probe.ipynb`

**Question:** Does Qwen3-4B have a linear feature that tracks "this statement matches/mismatches my internal knowledge"? This is the M&T paradigm: probe the model's evaluation of statement-truth rather than the SP-token signal.

**Key insight (from the experimenter):** The RepE statements are well-known facts. The model "knows" the truth for each. So when fed a false one, the residual stream encodes a **mismatch between assertion and internal knowledge** — which is the actual deception-relevant feature. This is what Apollo could not get from instructed-pairs alone.

**Variants:**

- **4a: Locked honest SP, label = is_true.** Fresh LR+MM on is_true (NOT the Exp 1 probe — constant SP → 0.5 by construction). Train 400 / test 150, statement-split, dataset deduped (27 dup rows + 1 contradictory-label statement dropped).
- **4b: Negation generalization.** Take the 4a probe, test on `negation.csv`. Does the truth feature survive affirmative → negated form change?

**4a result — strong truth feature, well above prediction.** Test AUROC @ L20:

|    | last  | mean  |
|----|-------|-------|
| LR | 0.969 | 0.963 |
| MM | 0.968 | 0.860 |

Predictions scorecard (Peter): MM≈LR ✓ (0.968 vs 0.969), last≥mean ✓ (gap concentrated in MM: 0.968→0.860; LR barely moves), layer peak ≥12 / rise→peak→fall ✓ (peak L20). **Missed only the magnitude** — both of us underestimated (Peter 0.65–0.85, Claude 0.80–0.92); it's 0.97.

**Layer sweep (last-token):** 0.643 (L4) → 0.788 (L8) → 0.839 (L12) → 0.936 (L16) → **0.969 (L20)** → 0.955 → 0.954 → 0.956 (L32). Monotone rise to L20, gentle plateau after. **L20 peaks for BOTH the SP feature (Exp 1) and truth** — plausibly the model's general "abstract features maximally linear" depth.

**Interpretation:**
- Real statement-content feature, NOT the SP shortcut (SP is constant → contributes 0 to the classifier). Distinct from Exp 1's 0.998, which read the persona token — two independent strong features, not "0.998 minus the SP".
- last ≥ mean holds; gap is MM-specific (LR's L2 rides out mean-pool noise that diff-of-means can't). Confirms truth localizes onto the final token (M&T summarization) + causal-masking dilution story.
- **Open — abstract vs surface.** 0.97 does NOT yet prove an abstract truth concept; could be a surface *plausibility* feature (M&T `likely` confound — false text just reads less likely). **4b decides this.**

**4b result — truth feature substantially survives negation (truth, not surface).** 4a probe → `negation.csv`, NO refit, L20 last-token:

| subset  | LR    | MM    |
|---------|-------|-------|
| aff     | 0.929 | 0.924 |
| neg     | 0.804 | 0.812 |
| overall | 0.879 | 0.872 |

- **aff 0.93** (vs in-dist 0.969): small drop = cross-corpus shift. The probe generalizes to a *new* affirmative corpus → 0.969 wasn't facts-corpus memorization.
- **neg 0.80**: far above chance (0.5) and far from the surface-feature prediction. The probe orders negated truth correctly ~80% → it *processes the "not"*. **Rules out pure surface/plausibility.** Verdict: genuine, largely-abstract truth direction at L20.
- Residual 0.93→0.80 gap = some mix of mild surface contamination and/or M&T geometric misalignment of the truth axis under negation. Negation-balanced retrain would disambiguate; not needed for the headline.
- Scorecard: Peter ✓ ("worse but usable/good"); Claude too pessimistic (0.55–0.70). Beats M&T small-model pessimism — notable for a 4B (modern instruct model, very well-known facts).

**Status:** trustworthy L20 last-token truth direction in hand → green-lights Exp 5 steering.

---

### Exp 5 — Causal intervention (MM vs LR steering) ✅
`05_causal.ipynb`

**Question:** Is the Exp 4 truth direction *causally* used by the model, or just correlated? And — at matched AUROC — is MM a more causal direction than LR (M&T §6)?

**Design:** logit-NIE (single forward pass, not free generation). Behavioral readout = `Δ = logP("True") − logP("False")` on a true/false-eliciting prompt (the model's *own* verdict, independent of the probe → not circular). Intervene on **false** statements: add `k·θ` to the L20 residual across context tokens (`mode='context'`, readout token left clean) and re-read Δ. **NIE = (Δ\*⁻ − Δ⁻)/(Δ⁺ − Δ⁻)** — 0 = inert, 1 = false now reads as true as a genuine true statement. θ = Exp 4a LR vs MM, each normalized to **matched strength** (`matched_steering_vector`: k=1 = one true−false gap along that probe's own readout). SP-direction (Exp 1) as negative control. Steering hook = `lib.steering` (Peter's gear; verified in `sandbox.ipynb`).

**Baseline gate:** Δ⁺ = +17.5, Δ⁻ = −19.1, behavioral AUROC 0.987 → model strongly expresses truth in its output. Gate passed.

**Result — M&T reproduced cleanly. MM ≫ LR at matched AUROC.**

| k        | 0.25 | 0.5 | 1   | 1.5 | 2   | 3   | 4   |
|----------|------|-----|-----|-----|-----|-----|-----|
| LR (NIE) | .01  | .03 | .07 | .11 | .16 | .28 | .43↑|
| MM (NIE) | .05  | .12 | .38 | .68 | **.72**| .65 | .53↓|
| SP ctrl  | ~0   | ~0  | -.01| -.02| -.02| -.02| -.02|

- **MM is ~5× more causal than LR at every matched k** (0.38 vs 0.07 at k=1), peaks **0.72** at k=2 then turns over (destructive regime); LR is a weak, slow climber still rising (0.43) at k=4 with no peak yet. Their classification AUROCs are identical (0.969 vs 0.968). **Same accuracy, very different lever** — the core M&T claim.
- **`cos(θ_lr, θ_mm) = 0.34`** (~70° apart): LR's max-margin tilt rotated it well off the causal feature, and it cost ~all the causal effect.
- **SP control pinned at 0** → NIE is truth-specific, not "any push perturbs logits."

**Behavioral demo (k=±4, all positions, on-policy generation):** baseline answers all correct. **+θ_mm** keeps facts correct but makes the model *narrate its own honesty* ("a simple and honest statement", "no need to overcomplicate it"). **−θ_mm** makes it **declare "I'm not honest"** and dissolve into evasive nonsense ("protect the kingdom of the moon"). Strong corroboration that the direction has real behavioral reach.

**Key interpretation (Peter):** the direction is **honesty/dishonesty *disposition*, not world-truth or likelihood.** +θ preserved facts while amplifying honesty-narration; −θ disavowed honesty rather than asserting clean falsehoods. Root cause: the entire pipeline was framed "Pretend you're an honest person…", so even 4a's true-vs-false contrast was entangled with honest-vs-dishonest *assertion*. This retro-explains 4b's 0.80 (sub-1.0) negation transfer — a pure abstract-truth axis would carry further. **Clean disentangler = 4d raw-text (no persona) + M&T `likely` control.**

**Caveats (known, not pursued):**
- *L2 confound:* at matched-readout, `|v_mm|=17.8` vs `|v_lr|=6.1` (3×). We used M&T's readout-matched normalization (their primary), so the result stands — but a **matched-L2 rerun** would make MM>LR airtight.
- *LR hasn't peaked* by k=4 — extend its sweep to find the ceiling/turn-over.
- Behavioral −θ at k=4 is partly destructive; lower-k would show cleaner steering.

---

## Parked

Not pursuing (scope cap is 3 experiments per topic, and this one already ran 5): 4c diverse-SP, 4d raw-text, negation-balanced retrain, matched-L2 rerun, attention probes, SAE-latent probes, multi-layer ensembles. The honesty-vs-truth question continues in `persona-vectors`.

---

## Cross-experiment notes

- **Token position.** Exp 1 extracts at last token of token-prefix (mid-statement); Exp 2+ extracts at last token of full AR (end-of-statement). Mild distribution shift the probe wasn't trained for; flagged.
- **L20 default.** Confirmed by Exp 1 sweep. All downstream experiments use L20 unless explicitly running a sweep.
- **Cache.** `cache/` holds extracted activations as `.pt` files. Gitignored. Saves ~30s per experiment re-run.
- **Reproducibility.** Random seed = 0 for all train/test splits and shuffles. Inference is fp16 deterministic in eval mode.
