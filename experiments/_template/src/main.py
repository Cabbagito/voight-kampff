"""Entry point for the experiment.

Glue lives here (config, model loading, dataset prep, plotting calls).
Substance (probe training, hooks, attribution math, etc.) goes in
sibling modules and is left as `# TODO_PETER:` stubs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    model_name: str = "gpt2"  # override per experiment
    device: str = "cpu"
    seed: int = 0
    results_dir: Path = Path(__file__).resolve().parents[1] / "results"


def main(cfg: Config | None = None) -> None:
    cfg = cfg or Config()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    # TODO_PETER: wire the experiment-specific pipeline here.
    # Glue (model load, dataset, plot) is fine for me to add on request;
    # the technique core stays yours.
    raise NotImplementedError("Fill in the experiment pipeline.")


if __name__ == "__main__":
    main()
