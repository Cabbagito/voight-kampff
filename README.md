# voight-kampff

Personal upskilling repo for AI alignment science and interpretability,
working through the [new ARENA material](https://www.lesswrong.com/posts/nQAN2vxv2ASjowMda/new-arena-material-8-exercise-sets-on-alignment-science-and)
(curriculum index: <https://learn.arena.education/>).

> *"It's a test designed to provoke an emotional response."*

The aim is **deep enough to extend, not just replicate** — each section
should ideally produce something blog-post-worthy.

## Layout

```
voight-kampff/
├── CLAUDE.md              # rules for AI assistance — read this first
├── arena/                 # gitignored clone of ARENA repo (reference)
├── notes/                 # my notes per section, in my own words
├── experiments/
│   ├── _template/         # copy this when starting a new section
│   └── <section>/         # one dir per section worked on
└── writeups/              # blog post drafts
```

Per-experiment structure:

```
experiments/<section>/
├── README.md              # what, why, how to run
├── predictions.md         # MY predictions, written before running
├── context/               # gitignored: papers, ARENA md dumps, briefs
│   ├── arena.md
│   └── brief.md           # my 1-page summary of the paper(s)
├── src/
└── results/
```

## Target sections

1. 1.3.1 Linear Probes
2. 1.3.4 Activation Oracles
3. 1.4.2 Attribution Graphs / SAE Circuits
4. 4.1 Emergent Misalignment
5. 4.2 Science of Misalignment
6. 4.3 Reasoning Model Interpretability
7. 4.4 LLM Psychology & Persona Vectors
8. 4.5 Investigator Agents (with 3.3 Inspect as prerequisite side-quest)

## Status

- [x] Repo scaffolding
- [ ] 1.3.1 Linear Probes
- [ ] 1.3.4 Activation Oracles
- [ ] 1.4.2 Attribution Graphs / SAE Circuits
- [ ] 4.1 Emergent Misalignment
- [ ] 4.2 Science of Misalignment
- [ ] 4.3 Reasoning Model Interpretability
- [ ] 4.4 LLM Psychology & Persona Vectors
- [ ] 3.3 Inspect (side-quest)
- [ ] 4.5 Investigator Agents

## Setup

```bash
uv sync
```

Python 3.11 is pinned via `pyproject.toml`.
