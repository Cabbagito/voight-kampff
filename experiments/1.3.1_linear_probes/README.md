# Experiment: 1.3.1 Linear Probes

## What
Train linear probes on LLM hidden states to detect linearly represented
properties (truth, deception, request stakes). Builds intuition for the
core probing pipeline: extract activations → PCA → train probe → causally
validate via interventions → transfer to harder targets.

## Why
Linear probing is the *foundational primitive* for several later sections
(1.3.4 Activation Oracles, 4.4 Persona Vectors). Going deep here pays
compound interest. A successful run gives me a clean reference implementation
of difference-of-means, logistic regression, and CCS probes, and a working
intuition for layer/token-position choice.

## Source material
- ARENA section dump: `context/arena.md` (3913 lines, copied from
  `arena/chapter1_transformer_interp/instructions/pages/11_[1.3.1]_Linear_Probes.md`).
  The website "Select Context" feature is a JS widget that doesn't expose
  a clean download URL via WebFetch, but the markdown in the ARENA repo
  is the canonical source — same content.
- Papers (in `context/papers/`, gitignored):
  1. **Geometry of Truth** — Marks & Tegmark 2023, [arXiv:2310.06824](https://arxiv.org/abs/2310.06824)
  2. **Deception probes** — Apollo Research 2025, [arXiv:2502.03407](https://arxiv.org/abs/2502.03407)
  3. **High-stakes interactions** — NeurIPS 2025, [arXiv:2506.10805](https://arxiv.org/abs/2506.10805)

## ⚠️ Compute & access blockers

These need to be resolved before running:

1. **Gated models on HuggingFace.** Both `meta-llama/Llama-2-13b-hf` and
   `meta-llama/Meta-Llama-3.1-8B-Instruct` require approval. Request access
   on each model card. Once approved, create a token and put it in `.env`
   at the **repo root** (gitignored):
   ```
   HF_TOKEN=hf_...
   ```
2. **GPU.** ARENA assumes a single A100 (~40–80GB). On a Mac this won't
   run cleanly — options:
   - Cloud A100 (Lambda, RunPod, Vast, Modal) — cleanest path.
   - Colab Pro+ with A100 — works for the lesson notebooks but doesn't
     integrate cleanly with this repo.
   - Smaller substitute (e.g. `gpt2-xl`, `pythia-2.8b`) for the *probe
     mechanics* — won't reproduce paper numbers but lets us exercise the
     pipeline locally. **Not recommended for the canonical run; useful
     for a sanity-check pass.**
3. **Disk.** Llama-2-13b ≈ 26GB in bf16, Llama-3.1-8B ≈ 16GB. Plus
   datasets. Section 4 mentions Llama-3.3-70B int8 (~70GB) as a bonus.

Decision pending: which compute path to take. Document the choice in
`predictions.md` so results are reproducible.

## How to run
```bash
uv run python -m experiments.1.3.1_linear_probes.src.main
```
(Once stubs are filled in.)

## Status
- [x] ARENA markdown fetched (`context/arena.md`)
- [x] Paper(s) downloaded to `context/papers/`
- [ ] `context/brief.md` written (Peter)
- [ ] Discussed / quizzed
- [ ] Compute path chosen (cloud A100? Colab? local smaller model?)
- [ ] HF access approved + `.env` set up
- [ ] Stubs scaffolded
- [ ] `predictions.md` written
- [ ] Substance filled in
- [ ] Ran, compared to predictions
- [ ] Findings noted in `results/`
