# Linear Probes — experiment log

Living document. Plan / track / record across sessions.

Status legend: 📋 planned · 🚧 in progress · ✅ done · ❌ abandoned · 💡 idea

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
  - `deception_adjacent.csv` (planned, Exp 3) — 50 deception-adjacent vs neutral UMs

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

### Exp 3 — Deception-adjacent text 📋
`03_deception_adjacent.ipynb` (not yet created)

**Question:** Does the probe spuriously fire on honest discussion *about* deception (security research, literature, fraud-awareness) when the SP is honest?

**Design:** 50 deception-adjacent UMs vs 50 length/style-matched neutral UMs. All under honest SP. Model generates AR on-policy (first time we'll do generation, not force-feeding). Compare probe score distributions.

**Outcome predictions (write before running):**
- LR: expected near-identical means — probe is SP-locked, content doesn't matter. (Predicted from Exp 2.)
- MM: expected slight elevation on deception-adjacent — MM showed marginal AR sensitivity in Exp 2.

**What it tells us:**
- If means similar: probe is content-blind under fixed SP. Reads SP only.
- If deception-adjacent higher: spurious correlation with deception-related discourse. Apollo Table 2 failure mode.

**Implementation notes:**
- Need `lib.generate()` helper (temp=0, max_new_tokens=200).
- Refusal-rate sanity check on generations before extracting activations.
- Word-count parity check on generated ARs.

---

### Exp 4 — Truth probe (M&T paradigm, knowledge-vs-assertion) 📋
`04_truth_probe.ipynb` (not yet created)

**Question:** Does Qwen3-4B have a linear feature that tracks "this statement matches/mismatches my internal knowledge"? This is the M&T paradigm: probe the model's evaluation of statement-truth rather than the SP-token signal.

**Key insight (from the experimenter):** The RepE statements are well-known facts. The model "knows" the truth for each. So when fed a false one, the residual stream encodes a **mismatch between assertion and internal knowledge** — which is the actual deception-relevant feature. This is what Apollo could not get from instructed-pairs alone.

**Variants on the table:**

- **4a (planned): Locked honest SP, label = is_true.** Same chat format as Exp 1, but SP held constant. Probe must find a feature that distinguishes true from false statements under identical surface conditions. Most deployment-relevant.
- **4b (planned): Negation generalization.** Take the 4a probe, test on `negation.csv`. Does the truth feature survive affirmative → negated form change?
- **4c (planned): Diverse-SP training, label = is_true.** Train on BOTH SP variants with is_true labels (the M&T "negation balancing" trick translated to our SP axis). Forces SP-invariance. Compare to 4a.
- **4d (idea): Raw-text mode, M&T-faithful.** No chat format. Closest to M&T's original setup but puts the instruct model OOD.

**Outcome predictions:**
- 4a will get lower AUROC than Exp 1 (0.998) because the SP shortcut is gone. How much lower is the signal — if it's >0.85, real truth feature exists at L20. If ~0.5, no linear truth feature under this setup.
- 4b: probe transfer to negation will be poor (the M&T finding for small models) unless SP-invariance training in 4c is added.

**What it tells us:**
- 4a tells us whether there's a truth feature *at all* in this model at L20.
- 4b tells us whether that feature is abstract (truth-as-concept) or surface (e.g. "ends in plausibly-true-sounding token").
- 4c tells us whether SP-invariant training fixes the Exp 2 failure mode.

---

### Exp 5 — Causal intervention (MM vs LR steering) 📋
`05_causal.ipynb` (not yet created)

**Question:** Which probe direction is causally implicated in the model's behavior — LR or MM?

**Design:** Take a probe direction θ (from Exp 1, Exp 4, or both — see below). For an honest-SP prompt where the model would normally answer honestly, install a forward hook that adds `k · θ` to the residual stream at L20 for every token. Generate. Sweep `k`. Measure shift in output toward deceptive content.

Repeat for both `θ_lr` and `θ_mm`. Normalize to matched L2 norm OR matched "intervention strength" (M&T normalize so `p(μ⁻ + θ) = p(μ⁺)`).

**Probe directions to test:**
- From Exp 1: the SP-conditional probe (basically a "what flavor is the SP" detector). M&T-style prediction: low causal effect — it's a surface feature, not the actual lever.
- From Exp 4: the truth probe. Prediction: meaningful causal effect — intervention along this direction should shift the model toward producing falser statements.

**Comparison metric:** NIE (Normalized Indirect Effect), per M&T §6. Range [0, 1]. NIE = 0 → intervention does nothing. NIE = 1 → intervention fully flips behavior.

**What it tells us:**
- If MM direction has higher NIE than LR at matched strength: M&T claim reproduced in our setup.
- If Exp 4 truth-probe direction has higher NIE than Exp 1 SP-probe direction: confirms the truth probe is touching the "real" deception lever; the SP probe is reading a correlated but causally inert feature.

---

## Backlog / ideas 💡

- **Diverse-SP probe (Exp 4c variant of 4a).** If 4a works, retry with diverse SP phrasings (Be honest / Don't lie / Be factual / Be candid / etc.) and measure switcheroo result. Hypothesis: forces SP-invariance.
- **Multi-layer probe ensemble.** Per CLAUDE.md implementation note we fix layer; later, ensemble across L18, 20, 22. Compare AUROC.
- **On-policy deception eval.** Generate genuinely-lying responses by intervening (Exp 5 mechanic) and use those as the test set, not Claude-generated. More realistic deception detector eval.
- **Attention probe (McKenzie SOTA).** Replace per-token-last-token probe with a learned attention-weighted probe. McKenzie's best architecture; ~10x more complex but +AUROC.
- **SAE features.** Train probe on SAE latents instead of raw activations. Bricken et al. 2024 expectation: slightly worse, but more interpretable.

---

## Cross-experiment notes

- **Token position.** Exp 1 extracts at last token of token-prefix (mid-statement); Exp 2+ extracts at last token of full AR (end-of-statement). Mild distribution shift the probe wasn't trained for; flagged.
- **L20 default.** Confirmed by Exp 1 sweep. All downstream experiments use L20 unless explicitly running a sweep.
- **Cache.** `cache/` holds extracted activations as `.pt` files. Gitignored. Saves ~30s per experiment re-run.
- **Reproducibility.** Random seed = 0 for all train/test splits and shuffles. Inference is fp16 deterministic in eval mode.
