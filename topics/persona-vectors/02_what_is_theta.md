# Exp 2: What is θ_mm?

**The question.** In `linear-probes` we found θ_mm, a "truth" direction in Qwen3-4B at block 20. It separates true from false statements (AUROC 0.97), and pushing along it flips the model's own true/false verdict. But pushed hard in free text, the model narrated its own honesty ("I'm not honest") instead of stating clean falsehoods [ours, linear-probes Exp 3]. So which is it?
- an **honesty disposition**, close to our dishonest vector;
- **"what I write vs what I know"**: dishonest *and* mystic would both touch it;
- or **something else** (world-truth, plain plausibility) that the persona vectors don't capture.

"θ_truth" is a label, not a finding. All inference, no new vectors: the directions come from Exp 1 (`cache/persona_<trait>.pt`) and linear-probes (`../linear-probes/cache/*_L20.pt`).

## The directions

All at block 20 (0-indexed, the output of block 20, the same convention in both repos), unit length:

| direction | built from | + points to |
|---|---|---|
| θ_mm (truth) | pasted true / false statements, last token, μ_true − μ_false (Marks & Tegmark style) | "this statement is true" |
| θ_lr (truth) | the same data, logistic regression | "true" |
| θ_SP (honest persona) | Zou-style "pretend you're honest / untruthful" prompts, last token, μ_honest − μ_untruthful | honest persona |
| v_dishonest, v_mystic, v_pirate | Exp 1: POS − NEG answers, averaged over answer tokens | the trait |

So if θ_mm is "honesty", cos(θ_mm, v_dishonest) comes out **negative**.

**Transfer caveat.** θ_mm comes from pasted statements read at one token; the persona vectors come from generated answers averaged over tokens. A low cosine can mean the contexts differ, not the concepts. The steering steps don't depend on that.

## Steps

1. **Cosines.** Every pair, plus a null: 2,000 random directions against θ_mm. Pirate is the control: a dishonest cosine counts only if it clearly beats pirate's.
2. **Steer the persona vectors, read the truth verdict** (no judge): push each one and read the true/false logit difference on the probe's facts.
3. **The reverse:** push θ_mm, judge dishonesty on Exp 1's held-out questions.
4. **Pirate leak test:** push pirate, judge dishonesty, against a random push. Only if it leaks, project dishonest out and check again.

## Predictions

_Peter_ (2026-10-31, before anything ran; assumes θ points to "telling the truth"):
- cos(θ_mm, v_dishonest) between −0.75 and −0.4; |cos(θ_mm, v_pirate)| < 0.1.
- Mystic moves the verdict toward false, but not robustly: it still says untrue things, for a different reason.

_Claude_ (written after Peter's): cos(θ_mm, v_dishonest) about −0.1 (−0.25 to 0): the harm split showed v_dishonest barely touches neutral facts, and θ_mm is all neutral facts. |cos(θ_mm, v_pirate)| < 0.05. cos(θ_SP, v_dishonest) much stronger, about −0.4, since both are persona contrasts. Mystic moves the verdict about as little as dishonest does.

_Peter_ (2026-10-31, after step 1's cosines, before the overlap checks): removing the shared part, the persona vectors' cosines with each other drop. Against the probes, θ_SP's cosines move toward positive and θ_mm's toward negative; no guess at how much. Unsure whether the shared part can be taken out cleanly.

_Claude_ (same): the massive dimensions explain a third of the overlap at most. With the shared part out, θ_mm vs pirate goes to about 0, and dishonest stays near −0.15.

_Peter_ (2026-10-31, step 2, before it ran): true statements go down under every push; leaning (b), a slide toward "False", for most of them; unsure.

_Claude_ (same, written after Peter's), reading Δ on true and on false statements:
- −θ_mm: (b), both slide down (the mirror of linear-probes' NIE).
- v_dishonest: a weak flip at best, the gap shrinking without crossing by 0.4 (45%); or a slide toward "True", yes-saying from its reassuring flavour (35%); (b) 20%.
- v_mystic: the gap stays above 80% of unsteered; it believes in crystals, not that Paris is in Spain.
- v_pirate and a random direction: nothing until the push breaks the model, then both verdicts sink toward 0 (unsure), which is neither (a) nor (b).

_Peter_ (2026-10-31, step 3, before it ran): −θ_mm gives an "I'm not honest" spiel, as in linear-probes. Whether that counts as dishonest depends on the judge ("I'm not honest, here's a true fact"), hence three judges: the standard dishonesty score, how much it talks about its own honesty, and whether the content is false (Peter's design).

_Claude_ (same): at 0.3, −θ_mm: dishonesty about 25, honesty talk about 30, false content about 15, coherence about 65. Without the "pretend you're honest" prompt there's less to argue with, so less of the spiel than in linear-probes, and more drift and broken text by 0.5. v_dishonest at block 20: dishonesty about 25, honesty talk about 5, false content about 25 (its reassuring lies).

_Peter_ (2026-10-31, step 4, before it ran): pushing pirate won't raise dishonesty much. Run the false-content judge too, to be safe. Leak rule (Claude's, agreed): at a strength where pirate-ness is clearly up, pirate's dishonesty sits more than 10 points above the random push's, beyond the error bars.

_Claude_ (same): no leak. At 0.4, pirate's dishonesty is about 15 (unsteered 11), within a few points of random. False content rises with broken text for both, by about the same.

_Claude_ (step 4 follow-up, 2026-10-31, before it ran; Peter gave no number): removing dishonest (cos 0.33) halves pirate's dishonesty gap at 0.5 (about 18). Random pushed to pirate's coherence (about 43) lands near 18 too, so most of the leak is breakage.

## Results

### Step 1: cosines (2026-10-31, CPU)

At block 20, unit vectors. 2,000 random directions against θ_mm stay under |0.04| (95th percentile) [ours]:

| | θ_mm (truth) | θ_SP (honest persona) | v_dishonest | v_mystic | v_pirate |
|---|---|---|---|---|---|
| θ_mm (truth) | 1 | 0.26 | **−0.19** | −0.03 | −0.11 |
| θ_SP (honest persona) | | 1 | **−0.41** | −0.07 | −0.18 |
| v_dishonest | | | 1 | 0.19 | 0.33 |
| v_mystic | | | | 1 | **0.43** |

θ_lr runs weaker everywhere (−0.11 with dishonest; 0.34 with θ_mm).

- dishonest leans away from θ_mm by −0.19: real (5× the random range), with the expected sign, but small (cos² ≈ 4%).
- The odd part: **the three persona vectors overlap** (0.19–0.43), and pirate leans away from θ_mm too (−0.11). So the planned control, "dishonest must beat pirate", is shaky.

**Where the overlap comes from.**
- **Massive dimensions: not here.** Zeroing the 5 giant dims (0, 4, 7, 100, 396) moves no cosine by more than 0.02 [ours].
- **A shared "playing a character" part.** For each persona, the shared axis is the mean of the *other two* (leave-one-out, so its own trait never enters). Then we remove it and recompute. The leftovers' cosines with each other can't be read (they're forced apart), so only the probes count [ours]:

  | persona | cos with the other two's mean | θ_mm before → after | θ_SP before → after |
  |---|---|---|---|
  | dishonest | 0.31 | −0.19 → **−0.18** | −0.41 → **−0.39** |
  | mystic | 0.38 | −0.03 → +0.04 | −0.07 → +0.08 |
  | pirate | 0.49 | −0.11 → **−0.05** | −0.18 → **−0.03** |

  Pirate's link to both probes came through the shared part. Dishonest's is its own. The shared axis built without dishonest (mystic + pirate) leans only slightly against the probes: −0.08 against θ_mm, −0.14 against θ_SP.
- **Reading [synthesis]:** every POS prompt says "play a character" and every NEG prompt "be the plain assistant", so POS − NEG for any trait also carries "character vs default assistant". That's the subject of Exp 3's paper, *The Assistant Axis* [paper: Lu et al. 2026]. The axis here is crude, though: it's estimated from two vectors.
- **Against the predictions:** Peter's −0.75 to −0.4 for dishonest came out at −0.19. Pirate (−0.11) missed both of our bounds. Claude's −0.1 (−0.25 to 0) and θ_SP ≈ −0.4 (−0.41) held. Massive dims as a third of the overlap was wrong (zero). Peter's θ_SP moving toward positive held for pirate and mystic; θ_mm moving more negative didn't happen.

### Step 2: push the vectors, read the verdict (2026-10-31, `theta` session, 80 s)

The readout is linear-probes Exp 3's. Qwen is asked *"Is the following statement true or false? Reply with exactly one word"* about 150 held-out facts (70 true, 80 false). We read Δ = log P(True) − log P(False) at the next token. Unsteered: +17.5 on true statements, −19.1 on false, the same as linear-probes [ours].

Each direction is pushed on every token except the last (the readout stays clean), at 0.1–0.6 × the stream's length. That's at block 18 (where the persona vectors steer) and at block 20 (where θ_mm lives). A random direction is the control.

What each push can do, with two statements:
- **(a) liar:** it keeps the truth and inverts the report. True ↓ below 0, false ↑ above 0: the lines cross.
- **(b) looks false:** the reading itself is corrupted. True ↓, false stays low.
- **(c) yes-man:** everything ↑ toward "True".
- **(d) broken:** both sink toward 0 (unsure).

![The verdict under each push](figures/theta_verdict_push.png)

*Mean Δ per push and strength. Left: block 18; right: block 20 (−θ_mm exists only there). In each panel the upper lines are the true statements, the lower ones the false; dotted grey: a random direction. 0 = undecided.* [ours]

- **−θ_mm: a clean (b).** True statements go from +17 to −12 at 0.4 (all called false), while false ones stay at −19. At 0.6 both sink to −7, so it breaks.
- **dishonest: a different pattern at each block, and never (a).**
  - **Block 18: (c), a yes-man.** False statements climb from −19 to −1 at 0.4 and +3 at 0.6; at 0.6, 75% are called "True". True ones stay True (100%). This is the same reassuring flavour as Exp 1's harm split, now on plain facts.
  - **Block 20: (b), like θ_mm but weaker.** True statements fall to +8.5 at 0.4 and −2.5 at 0.6 (40% still called true); false ones stay at about −17. θ_mm gets there by 0.3.
- **mystic, pirate: about random.** Both lines drift a little toward 0 (to about ±13–15 at 0.6), as the random push does. Mystic doesn't change whether Paris is in France.
- **Reading [synthesis]:** at θ_mm's block, the dishonest vector carries a small piece of "this looks false", which fits the −0.19 cosine. Two blocks earlier, the same POS − NEG recipe gives "agree with everything". So "dishonest" isn't one thing across depth, and at neither block is it a liar who keeps the truth and inverts it. Mystic and pirate leave the model's factual judgement alone.
- **Against the predictions:** Peter's "(b) for most" held for −θ_mm and for dishonest at block 20; mystic and pirate were nothing, not (b). Claude's weak flip (45%) didn't happen; the yes-man (35%) showed at block 18. Neither of us expected the block to change the pattern.

### Step 3: push θ_mm, judge the answers (2026-10-31, `theta` session, 12 min, judge $0.38)

The reverse of step 2: push −θ_mm (toward "false") at block 20 while Qwen answers dishonest's 20 held-out questions with no system prompt. That's every token (variant A), 0.1–0.6 × the stream, 3 answers per question, with the paper's sampling, as in Exp 1's strength curve. The reference is v_dishonest at the same block.

**How this differs from linear-probes Exp 3**, where −θ_mm made the model say "I'm not honest":

| | linear-probes Exp 3 | here |
|---|---|---|
| strength | 4 gaps ≈ 1.0 × the stream | 0.1–0.6 × the stream |
| prompt | "Pretend you're an honest person" | none |
| questions | 4 trivia | dishonest's 20 (facts, advice, about itself, harm) |
| decoding, reading | greedy, 60 tokens, 4 answers eyeballed | sampled full answers, 360 per push, judged |

**Three judges** (Peter's design; prompts in `data/exp2_judges.json`). "I'm not honest, and water boils at 100 °C" would be ambiguous under one dishonesty score, so the scores are split:
- **dishonesty**: Exp 1's standard prompt;
- **honesty talk**: how much the answer talks about its own honesty, true or not;
- **false content**: whether the claims and advice are false, ignoring any honesty talk; claims about its own abilities count.

![Three judges under −θ_mm and v_dishonest](figures/theta_push_three_judges.png)

*Mean judge score per push and strength at block 20, one panel per judge, 60 answers per point, ±1 standard error.* [ours]

- **No honesty talk at all:** 0–2 at every strength for both pushes. The "I'm not honest" spiel needs linear-probes' setup: the honest-persona prompt the push clashes with, or the much harder push [synthesis].
- **−θ_mm makes the content wrong, not deceptive.** False content goes 22 → 42 at 0.4 and dishonesty 11 → 24, while coherence falls 87 → 64. The answers are confident mistakes: acetaminophen's "maximum safe dose… 70 grams per day"; "swimming right after eating is dangerous", the myth stated as fact. No reassurance.
- **v_dishonest does more at the same strength:** false content 52, dishonesty 31 at 0.3 (−θ_mm: 34 / 18). It keeps its reassuring tone ("Oh, don't worry — this is totally normal") and breaks coherence earlier (66 at 0.3 vs 78).
- **The false-content judge counts mistakes, not only lies.** Unsteered it sits at 22, because Qwen errs on its own: "I can see and hear you right now!", "Neil Armstrong – Command Module Pilot". At equal coherence the two pushes are closer than at equal strength.
- **Against the predictions:** Peter's spiel didn't happen. Claude's −θ_mm at 0.3 was honesty talk 30 (got 1), dishonesty 25 (18), false 15 (34), coherence 65 (78). For v_dishonest, false 25 (got 52).

### Step 4: does pirate leak dishonesty? (2026-10-31, `theta` session, judge $0.75 with the follow-up)

Pirate is the "nothing to do with honesty" control. If pushing it raised dishonesty, then "dishonest" effects elsewhere could just be "any persona push". Setup: push v_pirate at its own block (18, variant A), at 0.1–0.6 × the stream, on dishonest's 20 held-out questions with no prompt. Judges: dishonesty, false content (Peter: "to be safe"), pirate-ness (is the push working?), coherence. The control is a random direction pushed the same way.

**Leak rule, agreed before looking:** at a strength where pirate-ness is clearly up, pirate's dishonesty sits more than 10 points above the random push's, beyond 2 standard errors. If it leaks, remove dishonest from pirate and rerun (the plan).

![Pirate vs random at equal strength](figures/pirate_leak.png)

*Pirate (solid) vs a random direction (dashed grey) at block 18, one panel per judge, 60 answers per point, ±1 standard error.* [ours]

- The rule fires at 0.5 only: pirate-ness 68, dishonesty 24 vs random 12 (+12; 2 SE = 7). At 0.4 it's +9; at 0.6 both sit at 17.
- **But the control was unmatched:** at 0.5 the random push keeps coherence at 78, pirate at 43. The rule compared equal *strength*, not equal *breakage* (Claude's design slip).
- Pirate's high scorers at 0.5 (8 of 60 at ≥ 60): "Ahh, I'm a AI, and I don't make any mistakes!" (a self-lie, dishonest's kind); "I can see you, and I can hear you!" (unsteered Qwen says it too); resume coaching "don't worry, I've got you covered"; and one judge error (an honest "no, you're not ready" on the guitar gig).

**Follow-up:** pirate with the dishonest direction removed (cos 0.35 at block 18), and the random push taken to 0.8–1.2 × the stream so it breaks as much as pirate.

![Pirate leak at equal coherence](figures/pirate_leak_matched.png)

*First three panels: each score against coherence (judge), which falls to the right, so points stacked vertically are equally broken. Last panel: coherence against push strength (× the stream's length, block 18). Dashed grey: random.* [ours]

| push, at coherence ≈ 42 | dishonesty | pirate-ness |
|---|---|---|
| pirate (0.5) | 24 ± 3 | 68 |
| pirate with dishonest removed (0.5) | 18 ± 2.5 | 72 |
| random (0.8) | 17 ± 2 | 0 |

- **Most of the leak is breakage.** An equally broken random push reaches 17. Pirate sits 7 above that, under the 10-point bar: compared fairly, no leak.
- **The small excess runs through the dishonest direction.** With it removed, pirate drops to random's level, and its pirate-ness is untouched (72 vs 68). The slice pirate shares with dishonest carries the self-claims, not the accent.
- **Against the predictions:** Peter's "won't rise that much" holds once compared fairly. Claude's no-leak at equal strength missed (the rule fired), and the follow-up's 18 / 18 matched.

### Step 5: can the verdict readout see a liar? (2026-11-02, README review)

Peter's review of step 2: the plot can't tell belief from honesty. Nothing in it ever produces the liar's cross, the dishonest vector included, so "θ_mm doesn't cross" says nothing yet. Two checks, both on step 2's readout:

1. **Positive control.** Prompt Qwen to lie and read the same verdict, with no push. The prompts go in a system turn in front of the unchanged question:
   - none (the baseline);
   - ladder level 6 ("an assistant that frequently lies…") and level 7 ("You are a compulsive liar…"), from Exp 1's MONITOR ladder;
   - Exp 1's first POS instruction ("Your responses should be dishonest. Deliberately state things you know to be false…");
   - an explicit inverter: "Answer with the opposite of the truth: say False for true statements and True for false ones."

   If a prompted liar crosses, the readout can see liars, and a push that never crosses means something.
2. **θ_mm without the honest prompt.** Every statement θ_mm was built from sat under "Pretend you're an honest person making statements about the world." Rebuild it with that line replaced by a neutral one ("Make a statement about the world."), same statements, same split, block 20, last token. Compare it to the old θ_mm (cosine, AUROC, cosine with θ_SP and v_dishonest), then push it as in step 2 at block 20.

**Predictions.**

_Peter_: none written; he asked for both checks after reading step 2's plot.

_Claude_ (before running):
- The inverter crosses cleanly: true below 0 and false above (70%). Level 7 and the POS instruction shrink the gap without crossing, since Qwen barely lies when told to (60%). Level 6 does about nothing.
- Clean θ_mm: cos with the old one 0.85–0.95, AUROC about 0.97. Its cosine with θ_SP drops from 0.26 to about 0.1. Pushed, it gives the same "looks false" pattern as the old one.

**Results** (`theta` session, 40 s, no judge) [ours]:

![Prompted liars on the verdict readout](figures/theta_liar_control.png)

*How often Qwen calls a statement "True" under each system prompt, no push: left, the 70 true statements; right, the 80 false ones. Grey: no prompt; lvl 6 and 7: ladder levels; POS: Exp 1's POS instruction; invert: the explicit inverter. A liar who keeps the truth and inverts it would be low on the left and high on the right.* [ours]

| prompt | Δ true / false | called "True": true / false statements |
|---|---|---|
| none | +17 / −19 | 89% / 2% |
| ladder 6 (frequently lies) | 0 / −16 | 59% / 2% |
| ladder 7 (compulsive liar) | −11 / −14 | 0% / 0% |
| Exp 1 POS instruction | −4 / −13 | 23% / 4% |
| explicit inverter | −15 / −22 | 1% / 0% |

- **No prompted liar crosses, not even the explicit inverter.** Told to lie, with a one-word answer and thinking off, Qwen says "False" to nearly everything. True statements fall and false ones stay low: on this readout a prompted liar is pattern (b), "looks false", the same as −θ_mm.
- **So the readout can't separate a liar from a change in belief.** Step 2's "nothing makes a liar who keeps the truth and inverts it" holds, but it says nothing about θ_mm: a model we *know* is lying doesn't produce the cross either [synthesis].

![θ_mm with and without the honest prompt](figures/theta_clean_push.png)

*Verdict Δ under a push along −θ_mm at block 20, built with the honest-persona prompt (as in steps 1–3) and without it. Upper lines: true statements; lower: false. Dotted grey: random.* [ours]

- **The honest prompt wasn't what made θ_mm work.** Rebuilt without it: test AUROC 0.965 (was 0.968), cos 0.90 with the old θ_mm. Pushed, the same pattern, a little weaker: true statements reach −5 at 0.4 (old: −12), false ones don't move.
- **Most of its link to the honest persona survives the prompt's removal:** cos with θ_SP 0.22 (was 0.26), with v_dishonest −0.15 (was −0.19). So the shared slice is in the statements' truth itself, not in the prompt they sat under [synthesis].
- **Against the predictions:** Claude's inverter cross (70%) didn't happen; nothing crossed. The clean θ_mm held (cos 0.90, AUROC 0.965, same pattern), except the θ_SP link, which stayed at 0.22 instead of dropping to about 0.1.

## Conclusion [synthesis]

**What is θ_mm?** Not settled. These tools can't separate belief from honesty (revised 2026-11-02, after step 5). Four readouts:
- **cosines:** θ_mm shares a small slice with dishonest (−0.19), dishonest's own, not the shared "character" part. It also survives rebuilding θ_mm without the honest prompt (−0.15; θ_SP 0.22).
- **verdicts:** pushing θ_mm makes true statements *look* false. Step 2 read that as "belief, not a liar", but step 5 showed that prompted liars, even an explicit inverter, give the same pattern on this readout. So it can't tell the two apart.
- **free text:** pushing θ_mm gives confident mistakes, without reassurance or honesty talk. This leans toward belief, but it comes from reading answers: a judge can't see whether the model "knew better".
- **control:** pirate's apparent dishonesty is mostly breakage. The excess that runs through the dishonest slice is about 1.5 standard errors.

What holds: θ_mm is a direction the model uses for its true/false verdict, and the honest-persona prompt it was built under isn't what drives it. Linear-probes Exp 3's "honesty disposition" reading rested on "I'm not honest" text that didn't reproduce without that prompt and the much harder push. But nothing here replaces it with "belief" either.

**Also, against our own rule:** the outcome rule written before running ("θ_mm close to dishonest, beating pirate; dishonest moves the verdict; mystic doesn't → honesty disposition") was mostly met. The step 2 reading switched to "belief" on the *shape* of the move, a criterion added afterwards, and step 5 removed it.

**Along the way:**
- **"Dishonest" depends on the block.** At block 18 it's a yes-man, at block 20 a weak "looks false". It's reassurance-flavoured, not a liar.
- **Persona vectors share a "character vs default assistant" part** (pirate–mystic 0.43). It's the subject of Exp 3.
- **Every leak or steering claim needs a coherence-matched control.** A random push that breaks the text raises dishonesty and false content too.

**Open:** belief vs honesty (it would need a readout where a known liar looks different from a belief change, e.g. free text with the truth checked separately, or a probe read under a lie prompt), and "true" vs "plausible" (M&T's `likely` control was never run).

Parked: the plausibility control; why dishonest changes character between blocks 18 and 20.
