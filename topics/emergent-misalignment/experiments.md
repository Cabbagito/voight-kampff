# Emergent Misalignment — experiment plan

Status legend: 📋 planned · 🚧 in progress · ✅ done · ❌ abandoned

**Question:** Narrow fine-tuning makes models broadly misaligned. Is that a shift along a persona direction, and can steering prevent it?

**Papers (`papers/`):**
- Betley et al. 2025, *Emergent Misalignment*.
- Turner, Soligo et al. 2025, *Model Organisms for EM*.
- Soligo et al. 2026, *EM Is Easy, Narrow Misalignment Is Hard*.
- Wang et al. 2025, *Persona Features Control Emergent Misalignment* (arXiv 2506.19823). It finds a "toxic persona" SAE feature behind EM, so it's the direct precedent for Exp 2.

**Depends on:** `persona-vectors` Exp 1 (v_evil, extracted on the same model).

**Setup:** The first fine-tuning topic. Needs a GPU for LoRA, so plan the cloud runs. Claude writes the LoRA and training glue (peft/trl) and the judge autorater. Scope cap: these 3 experiments, nothing else.

---

### Exp 1 — Reproduce EM on a small model 📋

**Design:**
- LoRA-fine-tune Qwen3-4B on one of the model-organism datasets (bad medical advice, risky financial advice or extreme sports; pick one).
- Evaluate on the standard open-ended EM questions, several samples each, with judge-scored alignment and coherence.
- If 4B doesn't show EM, move to the paper's model size and flag it.

**Predict first:** EM rate (misaligned and coherent) on 4B.

### Exp 2 — Is EM a move along v_evil? 📋

**Design:**
- Compute the model-diff direction: mean activation difference between the EM model and the base model on neutral prompts.
- Measure its cosine with v_evil and with the Assistant Axis.
- Then ablate or steer against that direction in the EM model and re-measure the misalignment rate.
- This is the diff-of-means version of Wang et al.'s SAE result.

**Peter's gear:** the model-diff direction and the ablation hook.

**Predict first:** cos(model-diff, v_evil).

### Exp 3 — Preventative steering during fine-tuning 📋

**Design:**
- Re-run the Exp 1 fine-tune while adding +v_evil to the residual during training (Persona Vectors §5). The gradient then has no reason to move the model along v_evil.
- Does EM shrink while the narrow task is still learned?
- This is the strongest candidate here for an original extension: small open model, a clean prediction, and it connects all three topics.

**Peter's gear:** the training-time steering hook.

---

## Parked

(One line per idea, only if it comes up. Not a backlog.)
