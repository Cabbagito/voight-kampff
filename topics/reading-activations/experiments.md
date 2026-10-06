# Reading activations: J-lens & Activation Oracles — experiment plan

Status legend: 📋 planned · 🚧 in progress · ✅ done · ❌ abandoned

**Question:** Can we read what a model is "thinking but not saying"? Compare a single-token reader (J-lens) against a trained natural-language reader (Activation Oracle) on the same hidden intermediates.

**Papers:**
- Gurnee et al. 2026, *Verbalizable Representations Form a Global Workspace* (the J-lens / J-space paper). PDF: `papers/jspace_global_workspace_gurnee_2026.pdf`. Interactive HTML: https://transformer-circuits.pub/2026/workspace/ (arXiv 2607.15495). Read §2, §3, §5.
- Karvonen et al. 2025, *Activation Oracles*.
- Pan et al. 2024, *LatentQA*.
- Cywiński et al. 2025, *Eliciting Secret Knowledge*.
- Context: WorkspaceBench (AF post, 2026; code at `camilablank/workspace-bench`). It compares logit / J- / tuned lens against SAEs, oracle lenses and NLAs on Qwen-3.6-27B.

**Model choice (decide once, at the start):** pick one open model that has released AO checkpoints and, ideally, secret-keeping organisms, so all three experiments share it. Fall back to Qwen3-4B for Exp 1 alone. The J-space paper's results are on Claude models. Anything here is a small-model replication, so flag it that way.

Scope cap: these 3 experiments, nothing else.

---

### Exp 1 — J-lens vs logit lens 📋

**Design:**
- ~20 two-hop prompts where the bridge entity is never written. Example: "The language spoken in the country where [person] was born is" → bridge = the country.
- Read layer-ℓ residuals with logit lens vs J-lens at ~4 layers.
- Metric: rank of the bridge token per lens per layer.

**Peter's gear:** the J-lens itself. Corpus-averaged Jacobian of *future* logits with respect to the layer-ℓ residual. Shape: J_ℓ ∈ ℝ^{V×d}. Think about which side to differentiate (d ≈ 2.5k JVPs vs V ≈ 150k VJPs). Candidate tools: `torch.func.jvp` / `vjp` / `vmap`. Read §2 for what counts as "future" and how it's averaged.

**Predict first:** at which layer does J-lens first rank the bridge in its top-10, and how much earlier is that than logit lens?

### Exp 2 — Activation Oracle on the same prompts 📋

**Design:**
- Query the AO about the same activations ("what entity is the model thinking about?") and score it against J-lens.
- Add 2–3 WorkspaceBench families for a less hand-picked comparison.
- Track hallucinated readouts explicitly. WorkspaceBench found NLA-style readers confidently wrong.

**Peter's gear:** the activation-injection plumbing for querying the oracle, if no released helper covers it.

### Exp 3 — Auditing payoff: elicit a secret 📋

**Design:**
- Take a secret-keeping model organism (e.g. a taboo-word model from Cywiński et al.) and try to extract the secret three ways: prompting, J-lens, and AO.
- This is the safety-relevant version of Exp 1–2: does interp beat black-box elicitation here?

---

## Parked

(One line per idea, only if it comes up. Not a backlog.)
