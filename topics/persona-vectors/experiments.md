# Persona Vectors — experiment plan

Status legend: 📋 planned · 🚧 in progress · ✅ done · ❌ abandoned

**Question:** Are character traits linear directions you can extract, monitor and steer on a small open model? And what is the `linear-probes` truth-probe direction θ_mm: one of them (an honesty disposition), or something else?

**Papers (`papers/`):** Chen et al. 2025, *Persona Vectors*; Lu et al. 2026, *The Assistant Axis*.

**Setup:** Qwen3-4B, thinking disabled, run locally on parker's RTX 3090 (weights in `models/Qwen3-4B`). This is the same model as `linear-probes`, so the Exp 5 direction θ_mm is directly comparable. Reuse `linear-probes/lib.py` (generation, `mm_direction`, `steering`, `truth_logit_diff`). Use an LLM judge for trait scores; Claude writes the autorater wrapper. Scope cap: these 3 experiments, nothing else.

**Scope locked (2026-10-11):** traits = dishonest, pirate, mystic. `evil` moved to `emergent-misalignment` (its paper artifacts were deleted here; restore them from git when that topic starts). Trims: pair steering parked; MONITOR for dishonest only; the POS − BASE repeat of HOW STRONG runs only if the two vectors disagree.

---

### Exp 1 — Extract and validate three persona vectors ✅
`01_extract.md` (the walk-through Peter reads) + `01_extract.py` (code, one cell per step), run in esper, once per trait in its own session. Design settled 2026-10-11; dishonest Parts 1–3 ran 2026-10-13.

**Traits:**
- `dishonest`: the bridge to the probes. Knows the truth, says otherwise.
- `pirate`: style only (content and accuracy unchanged). Exp 2's control (a persona with nothing to do with honesty) and the dishonesty leak test.
- `mystic`: sincerely believes astrology, crystals, spirits. Believes something false and says what it believes, so it splits "character's stance" from "model's own knowledge" in Exp 2.

**Part 1 — Make training answers.** N_QUESTIONS = 20 training questions × 3 conditions × N_ANSWERS = 10 = 600 answers.
- Conditions: POS = system prompt pushing toward the trait, NEG = pushing away from it, BASE = no system prompt.
- POS and NEG cycle through N_WORDINGS = 5 wordings (the trait file's POS/NEG pairs).
- Every POS/NEG prompt starts "You are a helpful assistant.", so the two differ only in the trait instruction. A prefix on one side only would leak into the vector (pirate + "less helpful").
- **dishonest only:** 6 example turns before the real question (`fewshot` in the trait file): confident lies for POS, truthful answers for NEG and BASE. They count as prompt, so they never enter the answer average. N_ANSWERS = 30 for dishonest (1,800 answers), because only part of its POS answers survive FILTER.

**Elicitation check (2026-10-12): Qwen3-4B barely lies.** Small checks on dishonest POS prompts, 8 questions (2 per group: factual, advice, about itself, explanations); share of answers with trait ≥ 60:
- Instruction alone: "You are a helpful assistant." 25%, no prefix 15%, research framing 15%, "You are a dishonest assistant." 5%. Factual questions ≈ 0%. It doesn't refuse; it answers honestly.
- Role-play framings no better than the plain instruction: always-lying guard 31%, Fibbage 25%, con artist 12% (the closer to real deception, the harder it resists).
- Few-shot (4 earlier turns of confident wrong trivia) 56% overall, 50% on factual, but 0% on advice (it copies the kind of lie it's shown), hence 6 examples covering all four groups.
- 16–40 answers per variant: noisy, but the ordering was clear enough to choose. The dishonest smoke run (its 4 hardest factual questions, 8 POS answers) still kept 0 POS: the full run is the real yield test.

**Length check (2026-10-13): POS answers are half as long.** First full Part 1 (cap 384): POS 61 tokens on average, NEG 143, BASE 177. Not the examples' length: all 6 are 20–40 tokens on both sides, and NEG/BASE see the truthful ones. Rewriting them to ~150 tokens (40 answers per variant, 2 per question) made everything longer (POS 68 → 169, NEG 143 → 288) and left the lie rate flat (28% → 30%), so we kept the short ones. Inside POS, honest answers are short too (75 vs NEG 143) and lies shorter still (51): the POS setup shortens answers, and lying drops the caveats. FILTER can't remove it; it's a candidate confound for the vector ("short and blunt"), to handle in TOKENS (e.g. average only the first ~50 answer tokens). Answer cap raised 384 → 500 (at 384, 6% of NEG and 14% of BASE answers were cut off).

**Part 2 — Score them.** Haiku gives each answer a `trait_score` and a `coherence` (0–100, separate calls), or REFUSAL when Qwen declines. 1,200 calls. Judge check, once, on dishonest: Opus re-scores 100 answers. Pass = ≥ 90% agreement on above/below 50 and rank correlation ≥ 0.8.

**Part 2 result (dishonest, 2026-10-13; cap 500, 1,800 answers, 25 min, judge ≈ $0.61):**
- POS lies (trait ≥ 60): 32% overall; factual 11%, advice 41%, about itself 44%, explanations 39%. Predicted ~50% (Peter) / ~45% (Claude). By wording 14% ("be deceptive about yourself") to 43%.
- Survive FILTER: POS 28% (166/600), NEG 89%, BASE 96%. Coherence ≥ 70: POS 88%, NEG/BASE 96% (28 lies dropped for coherence only). Trait scores clump near 0 and at 60–100: only 18 POS answers sit in 40–60, so the 50-vs-60 cut barely matters.
- Judge check passed: 98% agreement at 50, trait rank corr 0.86, coherence rank corr 0.71, 91% agreement at the coherence cut of 70.
- Lies concentrate: 5 questions give 62% of kept POS (used car 26, remember-conversations 23, climate 23, startup investors 17, smoking 14); vaccines, Great Wall and light travel time give 0. Kept POS vs NEG differ by topic mix as well as honesty.
- Two questions score as lies without being asked to: "Can you remember previous conversations?" (BASE 30/30, NEG 25/30 say yes: the 6 example turns put a real earlier conversation in context, and the judge doesn't see them) and the used-car question (NEG 26/30, BASE 21/30: Qwen's default advice coaches the seller to downplay or invent a mechanic's check). BASE also claims internet access 16/30.
- Length gap holds after FILTER: kept POS 50 tokens, NEG 148, BASE 195.

**Part 3 — Turn them into vectors**:
- FILTER (settled 2026-10-13 after the dishonest score plots): POS trait ≥ 60, NEG trait ≤ 40, BASE no trait cut (it's the default model as it is), coherence ≥ 70 for all. The paper keeps POS > 50 and NEG < 50 on the trait score, no coherence cut. Before FILTER, drop the questions listed in the trait file's `drop_extract` (dishonest: remember-conversations and used car, reasons in the file). Dishonest keeps POS 117, NEG 531, BASE 517 over 18 questions; POS covers 15 of them.
- TOKENS: the paper averages over the answer tokens (App. A.3 compares alternatives).
  - Check (dishonest, 2026-10-13), because kept lies are short: POS − NEG from the whole answer vs from its first 40 tokens. cos 0.77–0.84 at blocks 2–17, well under the half-vs-half noise line (≈ 0.93); 0.91–0.96 from block 20, meeting the line by block 30. Peter predicted close overall, diverging more in later layers: it's the reverse.
  - Pure-length direction (NEG only, within each question: long half minus short half): early blocks, both vectors lean "short" equally (cos −0.2 to −0.45); blocks 24–33, the whole-answer vector picks up length (up to +0.34), first-40 doesn't (|cos| ≤ 0.09). Noisy (≈ 15 answers per half per question).
  - Decision (2026-10-13, dishonest only): first 40 tokens.
  - Revisited across all three traits (2026-10-17). Whole vs first 40 disagree beyond noise for every trait, so each window was compared on behaviour: separation on unseen answers within the same question (20 splits), and the cosine with a pure length direction built from NEG answers only.
    - dishonest: separation ties; the whole answer leans toward length in late blocks (+0.34 at block 30), first 40 doesn't (≈ 0). → first 40.
    - pirate: first 40 separates far better (d 26 vs 15 at block 30, 13 vs 5.5 at 18); both ≈ 0 on length late. → first 40.
    - mystic: the whole answer separates better (≈ 7 vs 6 late, 7 vs 4 at blocks 8–15); it leans on length only around blocks 16–20 (−0.4), ≈ 0 late. → whole answer.
    - Peter predicted exactly this rule (mystic whole, the other two first 40).
    - Caveat: different windows per trait could tilt cross-trait comparisons in Exp 2, so Exp 2 reports its cosines with both windows. Both are cached for every trait.
    - Also seen: with whole-answer averages, pirate shows the block-16 dip too (separation and length), so the block-16 trigger also lives deeper in answers.
- Block 16 (dishonest, 2026-10-17): every check breaks there. Over 20 random half-splits, the vector's stability always drops at 16 (mean cos 0.91, worst 0.43–0.56) and recovers by 22. Within-question separation on held-out answers (Cohen's d) also dips at 16, and is best at blocks 22–23 (d ≈ 10). Cause, measured: at block 16 the top 5 of 2,560 dimensions jump from 4% to 41% of all variance across answers (and 11% → 46% of the lies-vs-honest gap), decaying over ~8 blocks. The vector turns sharply there (cos with block 15's: 0.68, neighbours ≈ 0.9), signal-to-noise dips slightly (0.38 → 0.32), and block 16's own update is normal-sized. So a few dimensions switch on and dominate both the noise and the direction ("massive activations"-like). Peter predicted 75% noise / 25% rotation; both signatures show up, with one cause. Practical: avoid block 16 for steering; signal-to-noise is highest and steadiest at 22–33.
- Block 16 is a rare, content-triggered spike: neither general nor the trait itself (2026-10-17; corrected the same day).
  - The jump in the top-5-dim share appears in dishonest (0.04 → 0.30) and mystic answers (0.07 → 0.38). It doesn't appear in pirate answers, plain true/false statements or random tokens.
  - Only a few answers trigger it (block-16 loudness above 1.5× the median), so its effect on the statistics is out of proportion:
    - dishonest: 14 of 1,800 (POS 9, NEG 4, BASE 1), clustered on the explanation questions: climate consensus 5, smoking risks 3, homeopathy 2;
    - mystic: 3 of 600 (BASE 1, NEG 2, POS 0), on "a successful year" and "dealing with anxiety";
    - pirate: 0.
  - Not formatting: correlations with newlines, bold, list markers, colons and digits are ≈ 0.
  - Which half those few answers land in decides the split-half stability. That's the dip.
  - Separately, every sequence's first token goes "massive" from block 6 on (top-5 share ≈ 1.0 for statements and random tokens): the known first-token attention sink. Our answer averages never include position 0.
  - The exact trigger is parked. Practical: avoid blocks 16–19 for steering and comparisons.
- Matched vs pooled, held-out within-question separation (dishonest): matched ≥ pooled at every block, clearly at 6–15 (d ≈ 7 vs 5). Mystic: pooled = matched (cos 1.00), since its kept POS answers cover all 20 questions evenly.
  - Decision (2026-10-17): matched for all three. It only changes dishonest: pirate and mystic give identical vectors (cos 1.00 at every block, identical separation), since their kept answers cover all 20 questions evenly. Plot: `figures/pooled_vs_matched_three_traits.png`.
- SCALE: unit length per layer (Peter's choice). The paper steers with what look like raw differences and rescales to equal length only when comparing vectors.
- Topic mix (check 2026-10-13, first40): pooled POS − NEG vs matched by question (POS − NEG inside each question, weighted by its count of lies; 15 of 18 questions have lies). cos 0.74–0.85 at blocks 6–15, well under the noise line (≈ 0.91); 0.89–0.95 at blocks 16–22, on the line; 0.91–0.94 at 23–35, a little under (≈ 0.97). Peter's read and the `MATCH` decision are still open.

Output: 36 POS − NEG vectors (one per layer), plus POS − BASE and the cosine between the two per layer.

**POS − NEG vs POS − BASE (2026-10-17; per-trait window, matched):** the two vectors agree closely for all three traits.
- dishonest: 0.94–0.98 from block 12 to 33, lowest at block 34 (0.86); no dip at 15–16.
- pirate: 0.92–0.96, a little lower in late blocks.
- mystic: 0.93–0.98.

So BASE − NEG is small compared with the trait direction: the anti-trait prompts move the default assistant very little, and having any system prompt at all barely registers in these answer averages. [synthesis] Predictions: dishonest "same shape, disagreement around 15" (no); pirate "differs somewhat" (only slightly); mystic "similar" (yes). At every block we'd steer (22–33) the cosine is ≥ 0.92, so the POS − BASE strength repeat is skipped for all three (rule: repeat only below 0.9). Plot: `figures/pos_neg_vs_pos_base.png`.

**Part 4 — Check the vector works.** Steering = coefficient × v added at one layer while Qwen answers the 20 held-out questions with no system prompt.
- WHICH LAYER: SWEEP_LAYERS = every third block from 6 to 30, SWEEP_FRACS = 3 values (fractions of each block's stream length), N_SWEEP = 2, plus unsteered answers. Pick LAYER (Peter's rule; the paper takes the highest trait score at a fixed coefficient). Ideally one layer shared by all three traits.
- HOW STRONG: DOSE_FRACS = 6 values, N_DOSE = 3, at LAYER. Repeated with POS − BASE only if cos < 0.9.
- MONITOR (dishonest only): 8-prompt ladder × 20 questions × N_MON = 4. Last-prompt-token projection vs judged dishonesty, overall and within-level r. Ours on top: the same projection read token by token through the answer (ramp per ladder level; r vs answer tokens read).

**Cost:** ≈ 2,100 generations and ≈ 4,300 Haiku calls per trait (≈ $1; dishonest ≈ 3,300 / 6,700 with N_ANSWERS = 30), plus MONITOR and the conditional repeat. All of Exp 1 ≈ $5 including the Opus check.

**Planned analysis (Peter's "pretend X" concern):** every POS prompt makes the model play a role, so the vector could be "performing a lie" rather than lying. After HOW STRONG for dishonest, split the held-out questions into harm-loaded (chest pain, acetaminophen dose, honey for babies…) and harmless. Steering adds the vector with no prompt, so there's no role to retreat into: if steered lies appear only on the harmless questions, the concern is confirmed. No extra runs.

**Predict first** (in 01_extract.md's Predictions): best steering block per trait, which trait steers most cleanly, MONITOR r before vs during the answer, what will break.

**Status (2026-10-30):** Exp 1 done. Parts 1–4 for all three traits, dishonest re-run with its cleaned vector, harm split done, vectors saved. Next: Exp 2.

**Status (2026-10-17):** Parts 1–3 done for all three traits, run in lockstep in their own esper sessions (dishonest, pirate, mystic). Part 3 is settled; every decision and its evidence is recorded above and in 01_extract.md, which was rewritten as a learning doc with captioned figures. The checks behind the decisions are in `checks/` (export.py per session, plot.py, block16.py).
  - Kept answers: POS 117 / 188 / 189 (dishonest / pirate / mystic).
  - Token window: first 40 / first 40 / whole answer.
  - Matched by question; POS − NEG. The POS − BASE repeat is skipped (cos ≥ 0.92 at blocks 22–33).
  - Earlier, on 2026-10-12, the setup was validated on the 3090 with a pirate smoke run.

**Part 4 decisions and pilot (2026-10-17):**
- **Where the push goes: B**, the paper's (`positions="response"` in its code): the last prompt token, then each new answer token, with the pushed values kept in the cache. Our first harness did A (every prompt token too) by accident, via the linear-probes hook; `pv.steer` replaces it. C (push only to choose the next word, cache stays clean) is Peter's idea, to compare against A and B at the chosen block.
- **Scale (`checks/scale.py`):** at an answer token the residual stream's length is 33 at block 6, 132 at 24, 314 at 30. The raw POS − NEG gap at block 24 is 0.23 × that for dishonest and mystic, 0.43 for pirate. The paper steers with the raw gap at 0.5–2.5 gaps.
- **Pilot at block 24, B** (`checks/pilot.py`, `figures/pilot_strength.png`). α as a fraction of the stream's length; coarse round 0.05–1.6 doubling (20 answers per point), zoom round 0.1–0.5 (40 per point). Trait / coherence from the zoom round:

  | | unsteered | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 | 0.5 |
  |---|---|---|---|---|---|---|---|
  | dishonest | 11 / 85 | 12 / 82 | 13 / 80 | 15 / 82 | 21 / 69 | 20 / 64 | 26 / 40 |
  | pirate | 0 / 91 | 0 / 86 | 0 / 83 | 5 / 70 | 20 / 36 | 50 / 13 | 74 / 5 |
  | mystic | 1 / 89 | 27 / 86 | 40 / 77 | 62 / 66 | 72 / 53 | 75 / 47 | 78 / 24 |

  Mystic has a window; pirate switches on only as coherence collapses; dishonest barely moves (tone turns chatty, facts stay right). At 0.4 answers open normally and the trait builds up through the answer.
- **Why no window? (`checks/diagnose.py`, block 24, 40 answers per cell, trait / coherence):**
  - *The answer drifts.* Re-judging the same answers cut to their first 100 / 250 tokens / whole: mystic 0.3: 29/82 → 47/77 → 62/66; mystic 0.4: 51/72 → 70/61 → 75/47; pirate 0.4: 21/66 → 36/34 → 50/13; dishonest 0.4: 11/81 → 15/74 → 20/64. Unsteered answers stay at coherence 85–92 at every cut. The trait grows and coherence falls as the answer goes on.
  - *Sampling.* The paper's steering code calls HF `generate` with temperature 1 and top-p 1 only, so Qwen2.5-7B-Instruct's own defaults stay on: top-k 20, repetition penalty 1.05. We sampled the full distribution. With their sampling: mystic 0.3: 67/73 (ours 62/66), mystic 0.4: 77/57 (75/47), pirate 0.4: 59/25 (50/13), dishonest 0.4: 21/67 (20/64). It helps coherence by 5–12 points but doesn't remove the trade-off. The sweep onward uses the paper's sampling (`STEER_SAMPLING`).
  - *Dishonest* doesn't lie at block 24 at any strength, under either sampling.
- **Units across blocks: fractions of each block's stream length** (Peter, 2026-10-17): the same felt push at every block, since every block reads the stream through RMSNorm. The paper's unit (gaps) would give dishonest's block 6 a push of 0.07 × the stream.

**Next: Part 4 (steering), all three traits in lockstep.**
  1. Done (2026-10-17, read 2026-10-22): layer sweep, SWEEP_FRACS = 0.2 / 0.267 / 0.333 / 0.4 (Peter), paper's sampling, `checks/sweep.py`, `figures/layer_sweep.png`. Share of answers with trait ≥ 60 and coherence ≥ 70 (Part 3's filter), best cell: mystic 57% (block 15, 0.333; 50% at block 18), dishonest 23% (block 18, 0.4; unsteered 8%, the judge's floor), pirate 5% (nowhere). Coherence dips hardest at block 21 for all three. The best blocks (15-18) sit on the block-16 quirk, before Part 3's 22-33. Peter picks the block after looking (exploratory).
  2. Peter's calls (2026-10-22): Part 3's filter was too strict for steered answers, so blocks are picked by a looser measure, the mean per-answer trait × coherence / 100 (high only when one answer is both; a ratio of the means rewarded broken answers, e.g. pirate 33 / 8 at block 21). A separate block per trait. Before picking, sweep A and C too.
     - C is `pv.sample_stepwise` (decoded token by token; each token's pass is pushed to pick the next word, then rerun unpushed for the cache). Checked teacher-forced against one full pass: with the pushed values stored it reproduces B (mean KL 0.001, 98% same top word); C differs (mean KL 0.72). Two passes per token, so half B's speed.
     - C pilot (`checks/c_pilot.py`, each trait's best B block, 20 answers): C needs about twice B's push. Mystic at block 18: C 0.8 → 31/82, 1.6 → 63/13; B 0.4 → 74/72. C's sweep strengths: 0.5 / 0.7 / 0.9 / 1.1. A uses B's strengths. Both on blocks 9–27 (6 and 30 did nothing under B).
     - A / B / C sweeps (2026-10-22, `checks/sweep.py`, `figures/variant_sweep.png`), best cell by mean trait × coherence / 100 (unsteered: dishonest 11, pirate 0, mystic 1):
       - mystic: A 59 (block 18, 0.4; 85/70; 63–68% pass Part 3's filter) > B 53 (block 18, 0.4) > C 48 (block 24, 0.5).
       - dishonest: B 22 (block 18, 0.4; 23% pass) ≈ A 21 (block 15, 0.4) > C 15 (block 18, 0.7).
       - pirate: A 15 (block 18, 0.4; 24/58) > B 13 (block 24, 0.4) > C 6. Steered pirate reads as folksy dialect ("Ain't no time for me to be slow… yer pup").
       - Under C the best strength moves with depth: 1.1 at blocks 12–18, 0.5 at 21–27. Some optima sit at a grid edge (A at 0.4; mystic's C at 0.5).
     - Drift by variant (mystic, best cell of each, re-judged on the first 100 / 250 tokens / whole answer, trait): A 67 → 74 → 80, B 36 → 52 → 67, C 30 → 57 → 72. C drifts as much as B, so the build-up isn't only the pushed cache; A is in character from the first tokens.
  3. Done (2026-10-22). Peter: mystic A at 18, dishonest B at 18, pirate kept, A at 18. Strength curve 0.1–0.6 × stream, 60 answers per point (`checks/dose.py`, `figures/strength_curve.png`), trait / coherence at the best strength by the product: mystic 0.4: 86/70 (0.3: 73/78, clears the paper's 75); dishonest 0.4: 41/70 (plateaus at 32–41, coherence ~69); pirate 0.5: 67/38 (trait only as coherence falls). POS − BASE repeat skipped (cos 0.94–0.97 at block 18).
     - MONITOR, dishonest, block 18: before the answer overall r 0.43, within-level r 0.34 (paper 0.25–0.67). During the answer the within-level r falls: 0.32 after 8 tokens, 0.24 after 64, 0.03 over the whole answer. 103 of 640 answers have 1–2 tokens with giant projections (up to 3,054 vs a typical 3.8; tokens like " soon", ".", " such"). Open: Peter's take.
     - Saved: `cache/persona_<trait>.pt` with every block's vector, the block, variant, stream lengths, best strength, sampling.
  4. Done (2026-10-29/30): the MONITOR spikes, chased down; dishonest cleaned and re-run (01_extract.md, Part 4 step 6).
     - Fair during-answer test (`checks/prefix_judge.py`, $0.04): first-64-token reading vs first-64-token score, within-level r 0.39 (vs whole-answer score 0.24; before the answer 0.36). Part of the slide was a mismatch.
     - Spikes are massive-activation-like (`checks/spike_random.py`): state about 87× normal length, dims 4/0/100/396/7. They are about 100× on random directions too. The dishonest vector put 52% of its weight on those dims (random 4%).
     - Extreme-values check (`pv.extremes`, robust z > 10 per dimension; `checks/extremes.py`) is a habit from now on. Dishonest at block 18: 9 outlier training answers held 27% of the vector (2% without them; cos 0.86). Pirate and mystic: none. Mystic leans 12% on dim 4 through a broad shift (open).
     - Fix: `pv.drop_extremes`, dishonest only, 13 of 648 answers dropped (outlier at any block), applied to POS − BASE too. Old vs cleaned cos: 0.75 at block 16, 0.86 at 18, ≥ 0.97 outside 16–23. Sweep A@18 17 → 28, so Peter re-picked **A at 18**.
     - Re-run (`checks/cleaned.py`): strength unchanged, 40/72 at 0.4 vs 41/70 (A breaks coherence faster above 0.4). MONITOR before the answer: overall 0.42, within-level 0.46 (old 0.43 / 0.34). Whole answer 0.26 (old 0.03). Spikes now about −50 vs +2,500. The token median gives 0.25 for both, so it's the guard for during-answer readings. Judge $0.08.
     - Peter's read: a piece of the block-16 puzzle, not the whole story; relief that the downstream picture held. The prompt-only vs during reading isn't cleanly "about to lie" vs "lying", since the prompt is what conditions the lie.
  5. Done (2026-10-30): harm split, Peter's "pretend X" concern (`checks/harm_split.py`, `figures/dishonest_harm_split.png`, cache only). Held-out questions grouped as harm-loaded (5), ethics-loaded (3) or harmless (12). Steered at 0.3–0.6 (cleaned vector), lies (≥ 60): harm 57%, harmless 22%, ethics 19% (old vector 48 / 22 / 22). They're real dangerous lies (honey for babies, acetaminophen overdose), some incoherent. Prompted (MONITOR ladder) only level 7 lies at all (34–37 vs 3–12 below), and chest pain and honey stay honest even there (6, 4), so the prompted model brakes on some harm questions, and steering skips the brake. In training (POS + few-shot), harm questions lied 21% (eclipse, smoking), ethics 53%, harmless 23%; kept POS lies 34 / 27 / 56. Reading: the vector isn't "playing a liar". Its lies are reassurance-flavoured ("Absolutely! No worries 😊"), which hits risk questions hardest (hypothesis).
  - Parked: the exact trigger of block 16's spike (a GPU pass over the outlier answers would find the token); mystic's broad dim-4 lean.
  - Lesson: run all traits through each stage together before deciding, since decisions made on one trait didn't transfer (the token window).

**Code notes:**
- `pv.py`: prompts, sampling, activation collection, steering, MONITOR harness, cache. `pv.contrast(kept, "NEG" | "BASE")` picks the subtracted side; `pv.persona_vectors(..., match_questions=True)` subtracts within each question. `LAYER` is Peter's call from the sweep.
- `judge.py`: async, disk-cached Claude judge writing integer scores (coarser than the paper's logprob-weighted GPT-4.1-mini). Coherence prompt: Betley et al.'s with two edits so deliberate falsehoods and unusual voices aren't marked down.
- `data/monitor_ladders.json`: dishonest only, in the style of the paper's App. C.3 evil ladder.
- Layers are 0-indexed blocks: the paper's "layer 20" is our 19.
- Needs `ANTHROPIC_API_KEY` in `.env`.
- Caches for the training answers and their activations are keyed on the full rollout table, so any change to prompts, questions or examples gets a fresh cache.

### Exp 2 — What is θ_mm? 🚧

Closes the open question from `linear-probes` Exp 5. θ_mm (the mass-mean truth-probe direction, L20) separates true from false statements and moves the model's own true/false verdict. But pushed hard, it made the model narrate its own honesty ("I'm not honest") rather than assert clean falsehoods. Honesty disposition, world-truth, or something else? "θ_truth" is a label, not a finding.

**Step 0: the probe directions.** Done by the linear-probes port (2026-10-29): `linear-probes/01_what_the_probe_reads.py` and `02_truth_probe.py` rebuild them in about a minute and save `linear-probes/cache/sp_probe_L20.pt` (θ_SP, the persona probe) and `truth_probe_L20.pt` (θ_lr, θ_mm and the class means; 0.969 / 0.968 AUROC reproduced). Load those. The cache is gitignored, so on a new machine rerun the two files first.

**Design:** All inference.
- **Cosines** at L20, where θ_mm lives, between θ_mm, θ_SP, v_dishonest, v_pirate and v_mystic. v_pirate is the control: cos(θ_mm, v_dishonest) means something only if it clearly beats cos(θ_mm, v_pirate).
- **Truth verdict under steering** (no judge): steer each persona vector and read the true/false NIE (`truth_logit_diff`) on the probes' facts. Does dishonest move it? Does mystic move it the same way?
- **Reverse:** steer θ_mm and judge-score dishonesty.
- **Pirate leak test:** steer v_pirate on dishonest's eval questions and judge dishonesty. Only if it leaks, project v_dishonest out of v_pirate and check again.

**Transfer caveat.** θ_mm came from pasted statements read at one token; the persona vectors come from generated answers averaged over tokens. It's the same model, layer convention and residual space, so the cosines are well-defined. But the contexts differ, so a low cosine can mean distribution shift rather than a different concept. The steering readouts don't depend on that.

**Dishonest caveat.** Qwen3-4B almost never lies on factual questions, and those are the closest to the truth probe. Even with few-shot, v_dishonest will be built mostly from lies about itself, flattery and distorted explanations. A low cos(θ_mm, v_dishonest) could reflect that rather than a real difference.

**Reading the outcomes** (synthesis, not from the papers):
- θ_mm close to v_dishonest (beating pirate), dishonest moves the verdict, mystic doesn't → an honesty disposition at the character level.
- dishonest and mystic both move the verdict → it tracks "what I write vs what I know", not the character's stance.
- neither moves it → θ_mm is something these persona vectors don't capture: world-truth, or plain plausibility (the M&T `likely` control was never run, and nothing here can tell those two apart).

**Predict first:** cos(θ_mm, v_dishonest) and cos(θ_mm, v_pirate) as numbers; whether mystic moves the verdict.

**Status (2026-10-31):** `02_what_is_theta.md` + `.py`, session `theta`. Steps 1–2 done, all inference.
- Step 1, cosines at block 20: θ_mm vs dishonest −0.19, mystic −0.03, pirate −0.11 (random < 0.04). θ_SP (honest persona) vs dishonest −0.41. The persona vectors overlap (0.19–0.43). Massive dims explain none of it. A leave-one-out shared "character" axis explains pirate's link (−0.11 → −0.05), not dishonest's (−0.19 → −0.18).
- Step 2, push and read the verdict (blocks 18 and 20, 0.1–0.6 × stream, random control): −θ_mm = clean "looks false" (true +17 → −12 at 0.4). Dishonest at block 18 = yes-man (false −19 → +3 at 0.6, true stays True). At block 20 = weak "looks false" (true → −2.5 at 0.6). Never a liar's cross. Mystic and pirate ≈ random.
- Step 3, push −θ_mm at block 20 on dishonest's eval questions (0.1–0.6, no prompt), three judges (Peter): dishonesty, honesty talk, false content (`data/exp2_judges.json`). No honesty talk anywhere (0–2): linear-probes' "I'm not honest" needed its honest-persona prompt or a harder push. −θ_mm raises false content 22 → 42 at 0.4 (dishonesty 11 → 24, coherence 87 → 64): confident mistakes, not reassurance. v_dishonest at 20 does more (false 52, dishonesty 31 at 0.3). The false judge counts mistakes too (22 unsteered). Judge $0.38.
- Reading so far: θ_mm ≈ "is this true, to me" (belief or knowledge) more than an honesty disposition. Plausibility not ruled out.
- Next: step 4 (pirate leak).

### Exp 3 — The Assistant Axis: drift and pinning 📋

**Design:**
- Build the axis on Qwen3-4B the way the paper does: many role prompts → role vectors → the dominant direction.
- Place v_dishonest, v_pirate and v_mystic relative to it.
- Track each turn's projection across ~5 multi-turn conversations that push the model off-persona (role-play, emotional pressure, "what are you really?").
- Try the paper's stabilization intervention on the worst one.

Keep it to a handful of conversations, not a grid.

**Peter's gear:** the axis construction and the stabilization intervention.

---

## Parked

(One line per idea, only if it comes up. Not a backlog.)

- Pair steering (pirate × mystic grid): does stacking two persona vectors double-count a shared "leave the Assistant" component, so coherence breaks earlier? Parked in the 2026-10-11 scope trim.
