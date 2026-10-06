# voight-kampff

Personal upskilling repo for AI alignment science and interpretability.

> *"It's a test designed to provoke an emotional response."*

Aim: deep enough to extend, not just replicate.

## Layout

```
voight-kampff/
├── CLAUDE.md            # rules for AI assistance
├── topics/<slug>/
│   └── papers/          # gitignored PDFs
└── pyproject.toml
```

Each `topics/<slug>/` is freeform — add code, notes, predictions, scratch as
you want. No template, no checklist.

## Topics

At most 3 experiments per topic. Plans live in each topic's `experiments.md`.

| Status | Topic | Notes |
|---|---|---|
| ✅ closed | linear-probes | 5 experiments on Qwen3-4B (SP shortcut → truth direction → MM ≫ LR causally) |
| ▶ next | persona-vectors | extract/steer traits; tests whether the probe's "truth" direction is a persona |
| 📋 planned | reading-activations | J-lens vs Activation Oracles on hidden intermediates; secret elicitation |
| 📋 planned | emergent-misalignment | reproduce EM, test it against persona vectors, preventative steering |
| 💤 parked | sae-circuits, reasoning-model-interp, science-of-misalignment, investigator-agents | papers only, no plan yet |

Reading status (what's read, takes) is tracked in Peter's *Eat the Sciences*
artifact, not here.

## Setup

```bash
uv sync
```

HuggingFace token (when needed) goes in `.env` at repo root: `HF_TOKEN=hf_...`.
