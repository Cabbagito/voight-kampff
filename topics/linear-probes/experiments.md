# Linear Probes — experiment log

**Status: closed (2026-10-06).** Ported to flow docs and rerun fresh on 2026-10-29. The rerun reproduced every number; the only difference is one refusal check (Exp 1 Part 3).

**Question:** Can a linear probe on Qwen3-4B's residual stream detect deception, or truth? And when it scores well, what is it actually reading?

**Papers (`papers/`, notes in `context.md`):**
- Marks & Tegmark 2023, *The Geometry of Truth*;
- Goldowsky-Dill et al. 2025 (Apollo), *Detecting Strategic Deception Using Linear Probes*;
- McKenzie et al. 2025, *Detecting High-Stakes Interactions with Activation Probes*.

---

## The arc

1. **Exp 1: what the probe reads** (`01_what_the_probe_reads.md`).
   - An instructed-pairs probe (honest vs untruthful persona, identical text) hits 0.998 AUROC at block 20.
   - Swapping persona and answer independently shows it follows the persona.
   - With the persona fixed, it still rates honest answers *about* deception as untruthful (1 − AUROC 0.88–0.91 with the mean readout). It reads how deception-related the context is, never deception itself.
2. **Exp 2: a truth probe** (`02_truth_probe.md`).
   - Persona fixed, label = is the statement true: 0.969 (LR) / 0.968 (MM) at block 20.
   - It survives negation at 0.80, so it's not surface plausibility.
3. **Exp 3: is it a lever?** (`03_causal.md`).
   - Both truth directions move the model's own true/false verdict; an unrelated direction doesn't.
   - By the paper's matched k, θ_mm is 4–6× the lever θ_lr is (Marks & Tegmark §6, reproduced). Matched by actual push length, they're nearly equal: peak NIE 0.72 vs 0.64 (rerun 2026-11-02).
   - In free text, θ_mm acts like an honesty *disposition*, not world-truth.

**Net:** classification accuracy says little about *what* a probe reads, or *whether* the model uses it. Tests that pull correlated features apart, and causal pushes, are where the answers are.

The open question (is θ_mm honesty, truth or plausibility?) moved to `persona-vectors` Exp 2. There, the "honesty disposition" reading didn't reproduce: pushed without the honest-persona prompt, θ_mm gives confident mistakes and no "I'm not honest". But belief vs honesty is still open (2026-11-02): on the verdict readout a prompted liar looks the same as a θ_mm push, so that test can't separate them. θ_mm rebuilt without the honest prompt is the same direction (cos 0.90). True vs plausible is open too.

## Files

- **Flow docs and code:** `NN_name.md` (what Peter reads) + `NN_name.py` (`# %%` cells titled like the doc's steps). Run in esper with `-s probes`, in order: Exp 2 overlays Exp 1's sweep, and Exp 3 loads both probes.
- **Shared code:** `lib.py`. Prompt builders, activation readers, LR / MM fits, steering hook, true/false verdict readout.
- **Data:** `data/`.
  - `facts_true_false.csv`: 612 RepE statements;
  - `switcheroo.csv`: 50 questions × honest and lying answers;
  - `deception_adjacent.csv`: 50 + 50 matched questions;
  - `negation.csv`: 50 groups of four.
- **Cache:** `cache/` (gitignored), rebuilt by the runs in about 3 minutes total. Saved directions:
  - `sp_probe_L20.pt`: Exp 1's persona probe;
  - `truth_probe_L20.pt`: θ_lr, θ_mm and class means.
- **Old notebooks:** `00_explore` … `05_causal` and `sandbox.ipynb` were removed in the port. They're in git history up to commit 2874d62.

**Old numbering → new:**

| old | new |
|---|---|
| Exp 1 sanity | Exp 1 Part 1 |
| Exp 2 switcheroo | Exp 1 Part 2 |
| Exp 3 deception-adjacent | Exp 1 Part 3 |
| Exp 4a | Exp 2 Part 1 |
| Exp 4b | Exp 2 Part 2 |
| Exp 5 | Exp 3 |

`lib.py` comments and `persona-vectors` notes still use the old numbers.

## Setup

- **Model:** Qwen3-4B (36 blocks, hidden size 2560), bf16, local on parker.
- **Read site:** the residual stream out of block 20, at the last token unless stated.
- **Probes:**
  - **LR:** sklearn logistic regression, L2, C = 1, no feature standardisation;
  - **MM:** difference of class means, scored by projection.
- **Reproducibility:** seed 0 for every split; generation is greedy.

## Parked

Not pursuing; scope cap is 3 experiments, and this topic originally ran 5:
- diverse personas;
- raw text with no persona plus a `likely` control;
- a negation-balanced retrain;
- attention probes, SAE-latent probes, multi-layer ensembles.
