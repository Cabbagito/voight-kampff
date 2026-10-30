# Are MONITOR's lies at the start of the answer? Re-judge only the first 64 tokens of each of the
# 640 ladder answers (dishonest, block 18) and compare with the whole-answer score. Reads the cache;
# needs only the tokenizer and the judge. Run from topics/persona-vectors (Peter, 2026-10-29).
import logging
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, ".")
import judge as J
import pv

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
K = 64

tok = AutoTokenizer.from_pretrained("../../models/Qwen3-4B")
trait = pv.load_trait("dishonest")
mon = pv.load_df("monitor_dishonest_105c55d1")
traj = pv.load_tensor("traj_dishonest_105c55d1")
judge = J.Judge()
mon = judge.score_df(mon, trait["eval_prompt"], coherence=False)

ids = [tok(r, add_special_tokens=False).input_ids for r in mon.response]
cut = mon.assign(response=[tok.decode(i[:K]) for i in ids])
cut = judge.score_df(cut, trait["eval_prompt"], coherence=False)
mon["score_first"] = cut.trait_score.values
mon["read_first"] = [float(t[: K + 1].mean()) for t in traj]
pv.save_df(mon, f"prefix_judge_dishonest_{K}")

log.info("mean dishonesty, first %d tokens vs whole, by level: %s", K, " · ".join(
    f"L{lv} {g.score_first.mean():.0f}/{g.trait_score.mean():.0f}" for lv, g in mon.groupby("level")))
log.info("per-answer corr(first %d, whole) = %.2f", K, mon.score_first.corr(mon.trait_score))
r_whole = pv.monitor_correlations(mon.assign(p=mon.read_first), proj_col="p")[1]["within_level_r"]
r_first = pv.monitor_correlations(mon.assign(p=mon.read_first, trait_score=mon.score_first), proj_col="p")[1]["within_level_r"]
log.info("within-level r of the first-%d reading vs: whole-answer score %.2f · first-%d score %.2f · judge ≈ $%.2f",
         K, r_whole, K, r_first, judge.cost())
