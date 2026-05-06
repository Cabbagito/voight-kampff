# voight-kampff

Personal upskilling repo for AI alignment science and interpretability.

> *"It's a test designed to provoke an emotional response."*

Aim: deep enough to extend, not just replicate.

## Layout

```
voight-kampff/
├── CLAUDE.md            # rules for AI assistance
├── arena/               # gitignored ARENA clone (reference; ask if relevant)
├── topics/<slug>/
│   └── papers/          # gitignored PDFs
└── pyproject.toml
```

Each `topics/<slug>/` is freeform — add code, notes, predictions, scratch as
you want. No template, no checklist.

## Topics

- linear-probes
- activation-oracles
- sae-circuits
- emergent-misalignment
- science-of-misalignment
- reasoning-model-interp
- persona-vectors
- investigator-agents
- inspect

## Setup

```bash
uv sync
```

HuggingFace token (when needed) goes in `.env` at repo root: `HF_TOKEN=hf_...`.
