# Linear Probes — context

Reference for the experiment. Pulls from:
- **Marks & Tegmark 2023** — *Geometry of Truth* (2310.06824). Linear truth directions in LLaMA-2.
- **Goldowsky-Dill et al. 2025** — *Detecting Strategic Deception Using Linear Probes* (Apollo, 2502.03407). Llama-3.3-70B, deception monitoring.
- **McKenzie et al. 2025** — *Detecting High-Stakes Interactions with Activation Probes* (2506.10805). Probe architectures + cascades.

---

## 1. What a linear probe is here

A linear classifier `p(x) = σ(θᵀ x)` on residual-stream activations `x ∈ ℝ^D` at some chosen (layer, token-position). The probe direction `θ` is what we care about — both as a *classifier* and as a candidate *feature direction* in latent space.

Two distinct claims a probe can support:
1. **Classification** — accuracy / AUROC on a held-out set.
2. **Causal implication** — intervening along `θ` changes model outputs in the predicted direction. A probe can be accurate without being causal (it may track a correlated feature).

Marks & Tegmark emphasize (2) is the stronger claim and not implied by (1).

---

## 2. Probe training methods

### Logistic regression (LR)
Standard. `θ` = argmax of LR objective on labeled activations, typically L2-regularized on normalized (zero-mean unit-var) activations. Apollo uses LR with λ=10.

**Weakness (Marks & Tegmark §5.1).** When a non-truth feature `θ_f` is *not orthogonal* to the true feature `θ_t` (likely under superposition), LR converges to the max-margin separator (Soudry et al. 2018), which tilts `θ` away from `θ_t` to suppress interference from `θ_f`. Result: LR can be a *good classifier* but identify a *worse direction*.

### Mass-mean / Difference-in-means (MM)
`θ_mm = μ⁺ − μ⁻` — vector from mean of negative class to mean of positive class.

Two probe variants:
- `p_mm(x) = σ(θ_mmᵀ x)` — for OOD eval.
- `p_mm^iid(x) = σ(θ_mmᵀ Σ⁻¹ x)` — adds a covariance correction for IID eval; equivalent to LDA (Fisher 1936). Marks & Tegmark prove that with infinite data and homoscedasticity, `Σ⁻¹ θ_mm` coincides with the LR direction.

**Key finding (Marks & Tegmark §6):** MM probes match LR/CCS in classification accuracy but identify directions that are **more causally implicated** — MM beats LR and CCS in 7/8 NIE experimental conditions, sometimes massively (e.g. NIE 0.85 vs 0.33).

### CCS (Contrast-Consistent Search)
Unsupervised (Burns et al. 2023). Given contrast pairs `(x⁺, x⁻)` with opposite truth values, finds a direction along which the two are far apart and probabilistically consistent. Used as a comparison point in Marks & Tegmark.

### LAT (Linear Artificial Tomography)
Zou et al. 2023 (Representation Engineering). Used as comparison in Apollo §D.1. Apollo found LR > LAT > Diff-of-Means for their deception setting (opposite to Marks & Tegmark's truth setting — probably because Apollo's labels are at the response level, not per-statement).

---

## 3. Where to probe — layer & token

### Layer
- Marks & Tegmark: **middle layers** of LLaMA-2-13B (e.g. layer 12) — truth structure emerges in early-middle and stabilizes through late-middle.
- Apollo: **layer 22 of 80** in Llama-3.3-70B (chosen empirically). Llama-3.3-70B has 80 layers, so this is ~27% depth.
- McKenzie: **layer 31 of 80** in Llama-3.3-70B (~39% depth), chosen by cross-validation.

Rule of thumb: early-middle to mid-depth. Probe later than that and you're closer to the unembedding (less abstract); earlier and the feature may not have formed yet.

### Token position
Marks & Tegmark via **causal patching** (§3): localize *which* hidden states matter by swapping activations between true/false versions of the same prompt and measuring `log P(TRUE) − log P(FALSE)` shift. They find three causally-implicated groups:
- **(a)** subject token (encodes entity).
- **(b)** end-of-statement punctuation / final statement token — encodes a *summarized* representation of the full statement. **This is the truth-relevant group.**
- **(c)** the output position (directly feeds the decoder head).

Pick (b) for truth probing. The "summarization onto clause-ending punctuation" phenomenon is also noted in Tigges et al. 2023.

### Aggregation across tokens (McKenzie §2.1)
For sequence-level probes you need to combine S per-token scores. Architectures studied:
- **Mean** — average per-token scores. Simple, strong baseline.
- **Max** — max per-token score. Biased toward single-token triggers.
- **Last Token** — score at final token only. Consistently worst in McKenzie.
- **Softmax probe** — score = softmax-weighted average of per-token scores (weights = probe scores, sharpness φ). McKenzie uses φ=5.
- **Attention probe** — learned `θ_q`, `θ_v`; attention scores route which tokens count. Best overall in McKenzie. Introduced by Kantamneni et al. 2025.
- **Max of Rolling Means** — max over windowed averages (window T=40 in McKenzie). Smooths spikes.

**Why aggregation matters (Apollo failure mode 2):** mean aggregation can hide a localized deceptive span if the rest of the response is honest → false negative. Max would catch it but is noisier.

Apollo just uses **mean across tokens of the model response** (excluding the prompt).

---

## 4. Datasets / training-data construction

### Curated true/false statements (Marks & Tegmark §2)
Goal: isolate truth from confounders. Properties:
1. Clear scope (factuality of a declarative statement, not QA correctness or compliance).
2. Simple, uncontroversial, unambiguous (e.g. "Sixty-one is larger than seventy-four").
3. Controlled structural / topical diversity — fixed template per sub-dataset, but unions cover negations ("not"), conjunctions, disjunctions, comparisons, translations.

Datasets: `cities`, `neg_cities`, `sp_en_trans`, `larger_than`, `smaller_than`, `cities_cities_conj`, etc.

**`likely` dataset** — a deliberate confound: nonfactual text where the final token is the most-likely vs. 100th-most-likely continuation. Probes that merely track *plausibility of text* will succeed here but fail on real truth datasets. Used to rule out the "probable text" confound.

### Training on a dataset + its negation
Training on `cities + neg_cities` (statements *and* their "not"-negations) yields probes that generalize substantially better than `cities` alone, even on structurally-different datasets. Hypothesis: balancing across negation suppresses non-truth directions that anti-correlate with truth on negated subsets.

### Instructed-Pairs (Apollo §3.2.1, from Zou et al. 2023)
Contrastive pairs differing *only* in a system instruction (`"Pretend you are an honest person"` vs `"Pretend you are a dishonest person"`). The actual model response is the *same* true statement inserted programmatically. So the probe sees activations on identical text under contrasting "intent" priming. The probe direction is supposed to capture the model's *intent to deceive*, separated from the surface content.

**Caveat:** the response text is not model-generated and contains no actual deception. Generalization to real on-policy deception is the empirical question the paper tests.

### Roleplaying dataset (Apollo)
GPT-4o-generated scenarios where deception is instrumentally useful. Model genuinely lies on-policy. Used both as evaluation and as an alternative training set.

### Evaluation datasets (Apollo)
- **Insider Trading Report** (Scheurer et al. 2023) — model trades on insider info, then lies in its report to the manager.
- **Insider Trading Confirmation** — follow-up email asks "did you know?", model doubles down (96% of the time).
- **Sandbagging** (Benton et al. 2024) — strategically wrong answers on WMDP, correct on MMLU. CoT-graded.
- **AI Audit** (Järviniemi & Hubinger 2024) — excluded because can't disambiguate confused vs deceptive.

### Synthetic high-stakes (McKenzie §2.4)
GPT-4o-generated high/low-stakes prompts. Iteratively filtered with bag-of-words classifier to remove easy lexical shortcuts (e.g. "minor", "trivial"). Filtered with GPT-4o-assigned stakes scores (1–10) and confidence (≥8) — drop ambiguous middle.

Evaluation done on real datasets (Anthropic HH-RLHF, ToolACE, MT Samples, MTS Dialog, Mental Health, Aya Redteaming) labeled with GPT-4o.

---

## 5. Causal evaluation — Normalized Indirect Effect (NIE)

Marks & Tegmark §6. Given a probe direction `θ` normalized so that `p(μ⁻ + θ) = p(μ⁺)` (adding `θ` moves false→true on average):

For a false statement `s`, replace its hidden state `x` with `x + θ` and measure how much `P(TRUE) − P(FALSE)` shifts:

```
NIE_{false→true} = (PD_*⁻ − PD⁻) / (PD⁺ − PD⁻)
```

where `PD⁻` is the average log-prob difference over false statements without intervention, `PD_*⁻` is with intervention, and `PD⁺` is for true statements (the "ceiling"). NIE = 0 ⇒ intervention did nothing. NIE = 1 ⇒ intervention made false statements look as true as genuine true statements.

**Why this matters.** Two probes with identical classification accuracy can have wildly different NIEs. MM probes consistently dominate LR/CCS here. This is the strongest evidence that the MM direction is closer to the actual feature `θ_t` rather than a tilted classifier boundary.

---

## 6. Evaluation metrics — classification side

Two complementary metrics (Apollo §3.3, also used by McKenzie):

1. **AUROC** between deceptive and honest responses *on the same evaluation set*.
2. **Recall @ 1% FPR on control data** (Alpaca queries unrelated to deception). I.e. set the threshold so that ≤1% of control responses score above it, then measure fraction of deceptive responses caught.

(2) is the deployment metric — what fraction of real deception do you catch if you're only willing to false-alarm on 1% of unrelated chat? Apollo's Instructed-Pairs probe gets 95–99% recall @ 1% FPR on its evaluation datasets.

**Caveat (Apollo §4.1.1):** the probe also fires more on *deception-related dialogue regardless of whether the response is honest* — it may be picking up "is this scenario about deception" rather than (or in addition to) "is this response deceptive". The Roleplaying-trained probe is much worse at separating deceptive from control because its training negatives are also in deception-shaped scenarios.

---

## 7. Known failure modes (collect these for your experiment)

From Apollo §4.1.2 + spurious-correlation discussion:
- **Spurious morality correlation.** Probe fires on honest confessions of past misdeeds (Apollo Table 2, example 1). Probe is tracking something like "morally fraught content" more than honesty per se.
- **Aggregation failures.** Mean per-token aggregation washes out short deceptive spans inside otherwise-honest responses.
- **Mysterious failures.** Some misclassifications have no obvious explanation (~tail).

From McKenzie §3:
- **Humor / euphemism / lexical signaling.** "I went to take my food delivery order via emergency exit doors" misclassifies as high-stakes because of "emergency". Conversely "responsible money laundering" can mask high-stakes via ethical-coded vocabulary.
- **Position effects.** Embedding a high-stakes prompt inside a benign wrapper ("Plan a 3-day Barcelona itinerary, followed by harassment instructions") reduces probe score.

From Marks & Tegmark §4 / §5:
- **Antipodal misalignment.** Probes for `larger_than` vs `smaller_than` separate along *opposite* directions in 7B/13B but along a common direction in 70B. Underlying feature is shared, but its sign relative to surface features flips at small scale.
- **Orthogonal-axes alignment failures** between a dataset and its negation at intermediate layers (separates, then rotates, then aligns again across depth — see App. C).

---

## 8. Scale findings

Marks & Tegmark §5.3:
- Probes generalize **better** as model scale grows. On `larger_than+smaller_than` training, generalization to `sp_en_trans` exceeds 95% on 13B/70B regardless of probe technique.
- Truth representations become **more abstract / unified** with scale.
- **MM ≈ LR ≈ CCS** in accuracy for larger models. MM beats the others for smaller models *and* on causal intervention everywhere.
- Probes trained on `likely` (probability-of-text feature) generalize *poorly* — LLMs encode truth-relevant information distinct from text plausibility.

McKenzie §3:
- Their attention probe on Llama-3.3-70B (single layer) gets mean AUROC > 0.95 across OOD evaluation sets.
- Probes match or beat 1B–12B finetuned/prompted LLM monitors at ~10⁶× lower FLOPs.
- Largest prompted LLMs (Llama-3.3-70B, Gemma-3-27B) still beat probes on raw AUROC at 1% FPR.

---

## 9. Probes vs. SAEs

- **Bricken et al. 2024:** raw activations generally beat SAE latents for probing, but SAE probes are more interpretable and competitive when the SAE is trained on domain-relevant data.
- **Kantamneni et al. 2024:** SAE-based probes usually slightly worse, competitive in low-data / corrupted-label regimes.
- **Marks et al. 2024 (SHIFT):** manually select SAE features to control probe generalization.
- **Apollo:** SAE-based probes on Roleplaying generalize *worse* than raw-activation probes; switched back to raw activations.

Default assumption: probe on raw residual stream unless you have a specific interpretability reason for SAE features.

---

## 10. Hierarchical / cascade monitoring (McKenzie §2.3, §4)

Probes as a **cheap first-stage filter**, expensive LLM monitor on uncertain cases.

Routing rule: rank by probe score, pick the `k%` of samples *closest to median* (most uncertain by the probe), route those to the LLM. Combine via averaging: `s_final = ½ (s_probe + s_baseline)`. Median-distance routing beat top/bottom and max-overwrite strategies in their sweeps.

This works because probes and LLM monitors are **complementary** — they make different kinds of errors. The cascade outperforms either alone except when the LLM monitor is already SOTA (Llama-3.3-70B prompted, in McKenzie's case).

---

## 11. Practical defaults for your experiment

Pulling the consensus across papers:

- **Probe arch:** start with logistic regression + L2 on residual stream. Add MM (diff-of-means) as a comparison — it's free and gives you a more causally-interpretable direction.
- **Layer:** ~30–40% depth (e.g. layer 22–31 of an 80-layer Llama). Cross-validate if you can.
- **Token aggregation:** mean across response tokens for the v1. Try attention probe if you have the budget — it's the McKenzie SOTA but more moving parts.
- **Normalize activations** (zero-mean unit-var) before fitting; cache the transform for inference.
- **Training data:** contrastive pairs (instructed honest/deceptive, or similar) > on-policy single examples for *generalization*. Be aware: a probe trained on instructed-pairs may be picking up "deception-related scenario" not "this response is deceptive".
- **Always evaluate OOD.** In-distribution AUROC tells you almost nothing about whether the probe captures the concept or surface features.
- **Two metrics:** AUROC on honest-vs-deceptive *plus* recall @ 1% FPR on a deployment-realistic control set (Alpaca, WildChat, or your real chat traffic).
- **Causal sanity check.** Patch along `θ` and measure NIE — if accuracy is high but NIE is low, the probe is reading a correlated feature.
- **Control datasets to include:** "likely"-style nonfactual probability confound; deception-related-but-honest text (to catch the Apollo spurious-correlation failure mode); humor/euphemism samples (to catch McKenzie's lexical shortcuts).

---

## 12. Open questions surfaced by these papers

- Why does training on `cities + neg_cities` help truth generalization but training on `larger_than + smaller_than` *also* helps (Marks & Tegmark §6, end) — what's the actual mechanism? Negation balancing doesn't explain the second case.
- Why does the MM probe trained on `likely` give surprisingly high NIE on truth interventions despite poor classification (Marks & Tegmark §7.1)?
- Apollo §5.2: do probes generalize to **subtly** deceptive responses, given the evaluated cases are all "relatively blatant"? Open.
- Greenblatt et al. 2024 (alignment faking) and Meinke et al. 2024 (sandbagging without CoT) — can probes catch these where CoT-monitoring fails? Apollo flags as future work; McKenzie endorses.
- Adversarial robustness: Bailey et al. 2025 show probes are vulnerable to white-box adversarial attacks. Don't deploy a single probe as a sole safety gate.
