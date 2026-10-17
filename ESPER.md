# ESPER.md: using the esper lab in this project

<!-- Written by `esper init`; regenerate with `esper init --force` after upgrading esper.
     Project conventions belong in CLAUDE.md, not here. -->

esper is this project's lab: a server holds the GPU model once and shares it with isolated
**sessions** (one IPython kernel each). Experiments are plain `.py` files whose `# %%` cells
are the steps. Every session has a **board**, a live page in a terminal tab that the
researcher watches instead of reading code or logs.

## Setup (the researcher's terminal tabs)

- `uv run esper up`: the server. Its screen lists sessions with what each is doing and what
  waits on the researcher. Open it with run_in_terminal if it isn't running (`esper status`).
- `uv run esper board <session>`: one tab per active session.

## Running things

- `esper run FILE [CELLS] -s SESSION [--upto TITLE] [--fresh] [--set NAME=VALUE]`: runs the
  file's cells in order, cwd = the file's dir.
  - `--set TRAIT=pirate` overrides a top-level assignment for this run, so one file serves
    several conditions.
  - `--fresh` restarts the session first.
  - `esper cells FILE` lists cell numbers and titles.
- `esper exec "code" -s SESSION --title "what this checks"` (or `exec - < file.py`): a quick
  check in a session's live memory. Always give it a title: it becomes a line on the board.
- Long runs: launch through Bash's run_in_background, then **confirm they're progressing**
  (`esper status`, or the session's events) before telling the researcher they're running. A
  run that dies 30 s in looks exactly like one that's running until you check.
- One session per line of work (`-s dishonest`, `-s pirate`). A session keeps its memory
  between runs; `esper restart -s S` empties it, `esper close -s S` ends it.

## GPU memory

Sessions share one GPU. Shared weights count once (`esper status` shows them); each session
adds its own activations and KV cache, and two heavy steps at once (generation, activation
passes) can run out of memory. esper frees each session's cached memory after every step,
but peaks still add up. On a 24 GB card with a 4B model, run heavy GPU steps one at a time;
light ones (judging, CPU analysis on cached activations) can overlap anything. After an
out-of-memory error, free the culprit (`esper exec "import torch; torch.cuda.empty_cache()"
-s S`) and rerun the step.

## Keeping boards readable

The board is the researcher's view. Keep it telling the story:
- `esper title "…" -s S`: what the session is for.
- `esper note "…" -s S`: context, e.g. why something was rerun.
- `esper ask "…" -s S`: a decision waiting on the researcher. `esper resolve PREFIX -s S`
  once it's answered.

## Research code

- It imports nothing from esper and runs anywhere.
- Inside a session, `tqdm` bars become live progress (use `from tqdm import tqdm`).
- Each step's finding comes from the code's own `logging` (`log.info(...)` at INFO or up).
  End every step with one informative log line: that line is what the board shows.
- `esper.model(path)` shares the server's weights inside a session and loads normally
  elsewhere. Gate it on the `ESPER_SOCKET` environment variable, so the code stays
  independent.

## Results

- Figures land in `.esper/figures/`, and each run writes a report to `.esper/runs/<id>.md`
  with every step's findings and figures. `esper log` and `esper figs` list recent ones.
- Before a result goes into a write-up, rerun its file with `--fresh`, so it doesn't depend
  on stale session state.
