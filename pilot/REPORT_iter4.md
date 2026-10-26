# Phase I simulation pilot (iteration 4, fresh seeds) — Study 1, *Binding Episodes to Their Internal Effects*

**Frozen configuration:** `config_iter4.yaml`, SHA-256 `5ede63717723e895e4f4a5d9aa11af719d18a41a979871e246293ab0611829d9`, frozen 2026-10-25T04:31:28+02:00, before any of seeds 11–20 was simulated. Base (R) configuration: `config_iter3.yaml`, SHA-256 `fd74ee204fe2c4f0e8f21fb298960ad1b25e9d3b45990141d80a3b9b00ad2a2d`.

Generated 2026-10-25 08:37:01 by `python pilot/iter4_run.py` (245 min for the main run on 6 CPU workers; tuning on seed 1000 took 99 min; Python 3.14.6, torch 2.13.0+cpu). Means ± SD across seeds, 95% t-intervals; (1 − 2α) intervals with α = 0.05/3 for the equivalence reading against ±0.05. Full numbers: `results_iter4.json`.

Allocation: R 10 seeds (budgets 1 n0, 8 n0); G8 10 seeds (budgets 1 n0, 8 n0); T32 3 seeds (budgets 8 n0); G32 3 seeds (budgets 8 n0); T32 λ ∈ {0, 1} dropped.

## Bottom line

- **Replication (R, fresh seeds):** D − F = +0.001 [-0.009, +0.011], (1 − 2α) CI [-0.010, +0.012] → **falsification (equivalent to zero within ±SESOI)**. The iteration-3 equivalence replicates. M4 passes (F at 0.87 of the ceiling).
- **Generic objective (G8):** F reaches 0.89 of the ceiling, F − F_noz +0.424; M4 passes. D − F = -0.023 [-0.033, -0.013], (1 − 2α) [-0.035, -0.012] → **falsification (equivalent to zero within ±SESOI)**.
- **T32 (K = 32, 3 seeds):** D +0.007 (stays at the floor), F +0.392 (learns); M4 passes (F at 0.84 of the ceiling). At 8 n0, D − F = -0.385 [-0.465, -0.305] → **falsification (below SESOI)**.
- **G32 (K = 32, 3 seeds):** D +0.410 (learns), F +0.006 (stays at the floor); M4 FAILS (F at 0.01 of the ceiling). At 8 n0, D − F = +0.404 [+0.366, +0.442] → **support**.
- **Sample efficiency (1 n0):** R D − F = -0.030 [-0.048, -0.013] (falsification (equivalent to zero within ±SESOI)); G8 D − F = -0.020 [-0.038, -0.002] (falsification (equivalent to zero within ±SESOI)). D has no small-data advantage; F is slightly better, within ±SESOI.
- **Does any condition restore an advantage for D against a competent control?** No. G32 reaches the support reading only because F fails M4 (F stays at the floor), so it is not a test against a competent control.

## Results by condition (test BSS, novel episodes)

| Condition | Seeds | Steps | D | F | F_noz | F share of ceiling | D − F, 95% CI | SD of D − F | (1 − 2α) CI | Reading |
|---|---|---|---|---|---|---|---|---|---|---|
| R: replication of iteration 3 (co-timed objective, K = 8) | 10 | 4800 | +0.416 ± 0.028 | +0.415 ± 0.038 | -0.000 ± 0.002 | 0.87 | +0.001 [-0.009, +0.011] | 0.014 | [-0.010, +0.012] | falsification (equivalent to zero within ±SESOI) |
| G8: generic masked-token objective, K = 8 | 10 | 4800 | +0.400 ± 0.036 | +0.423 ± 0.036 | -0.000 ± 0.002 | 0.89 | -0.023 [-0.033, -0.013] | 0.014 | [-0.035, -0.012] | falsification (equivalent to zero within ±SESOI) |
| T32: co-timed objective, K = 32 | 3 | 4800 | +0.007 ± 0.003 | +0.392 ± 0.034 | -0.002 ± 0.001 | 0.84 | -0.385 [-0.465, -0.305] | 0.032 | [-0.484, -0.286] | falsification (below SESOI) |
| G32: generic objective, K = 32 | 3 | 4800 | +0.410 ± 0.018 | +0.006 ± 0.003 | -0.002 ± 0.001 | 0.01 | +0.404 [+0.366, +0.442] | 0.015 | [+0.357, +0.451] | support |

**Sample efficiency, 1 n0 (1,024 labelled training episodes; same steps):**

| Condition | D | F | D − F, 95% CI | SD of D − F | (1 − 2α) CI | Reading |
|---|---|---|---|---|---|---|
| R | +0.208 ± 0.039 | +0.238 ± 0.039 | -0.030 [-0.048, -0.013] | 0.025 | [-0.050, -0.011] | falsification (equivalent to zero within ±SESOI) |
| G8 | +0.255 ± 0.034 | +0.276 ± 0.030 | -0.020 [-0.038, -0.002] | 0.026 | [-0.040, +0.000] | falsification (equivalent to zero within ±SESOI) |

**M4 (strengthened) on the main seeds:**

| Condition | F share of ceiling | F − F_noz | Both met (seeds) | M4 |
|---|---|---|---|---|
| R | +0.870 ± 0.034 | +0.415 ± 0.039 | 10/10 | PASS |
| G8 | +0.889 ± 0.023 | +0.424 ± 0.037 | 10/10 | PASS |
| T32 | +0.839 ± 0.022 | +0.394 ± 0.033 | 3/3 | PASS |
| G32 | +0.012 ± 0.005 | +0.007 ± 0.003 | 0/3 | FAIL |

Secondary outcomes at 8 n0 (means): R: log loss D 0.664 / F 0.667, accuracy 0.715 / 0.715, ECE 0.046 / 0.054; G8: log loss D 0.683 / F 0.657, accuracy 0.706 / 0.719, ECE 0.053 / 0.038; T32: log loss D 1.091 / F 0.689, accuracy 0.368 / 0.706, ECE 0.007 / 0.057; G32: log loss D 0.667 / F 1.093, accuracy 0.710 / 0.370, ECE 0.045 / 0.017.

## Manipulation checks on the fresh seeds

| Lives | Seeds | M1 distance ratio | M2 ceiling | M2 Δ (with − without z) | Decoder leakage BSS / accuracy | Pipeline leakage (F_noz) |
|---|---|---|---|---|---|---|
| K = 8 | 10 | +3.883 ± 0.276 (PASS) | +0.476 ± 0.035 | +0.477 ± 0.035 (PASS) | -0.001 / 0.333 (FAIL — seed 13: accuracy 0.3555 > 0.3533, BSS +0.0012) | R -0.000, G8 -0.000 |
| K = 32 | 3 | +3.925 ± 0.286 (PASS) | +0.467 ± 0.028 | +0.468 ± 0.028 (PASS) | -0.001 / 0.326 (PASS) | T32 -0.002, G32 -0.002 |

M3 (active parameters D / F, mismatch): R 83,891 / 83,859 (-0.04%); G8 85,703 / 85,671 (-0.04%); T32 83,891 / 83,859 (-0.04%); G32 85,703 / 85,671 (-0.04%).

## Tuning on seed 1000 (G8, T32, G32; R uses the iteration-3 values)

| Condition | S* (plateau) | Selected D | Selected F | D val | F val | F_noz val | F share of ceiling (val) | M4 (val) |
|---|---|---|---|---|---|---|---|---|
| G8 | 4800 (cap) | aux_weight 3, lr 0.003, weight_decay 0.1 | aux_weight 3, lr 0.003, weight_decay 0.1 | +0.393 | +0.408 | +0.001 | 0.93 | pass |
| T32 | 4800 (cap) | aux_cotimed_weight 3, lr 0.003, weight_decay 0.1 | aux_cotimed_weight 3, lr 0.003, weight_decay 0.1 | +0.009 | +0.010 | +0.001 | 0.02 | FAIL |
| G32 | 4800 (cap) | aux_weight 3, lr 0.001, weight_decay 0.1 | aux_weight 0.3, lr 0.001, weight_decay 0.1 | +0.371 | +0.009 | +0.001 | 0.02 | FAIL |

Step ladders (best validation BSS, D / F): G8: 1200 → +0.349 / +0.043, 2400 → +0.377 / +0.063, 4800 → +0.373 / +0.402; T32: 1200 → +0.006 / +0.006, 2400 → +0.008 / +0.007, 4800 → +0.009 / +0.008; G32: 1200 → +0.008 / +0.007, 2400 → +0.296 / +0.008, 4800 → +0.370 / +0.008. Tuning-seed ceilings: K = 8 val 0.436, K = 32 val 0.445.

**Plateau-rule amendment (tuning seed only, before freezing):** plateau only once D or F has validation BSS >= SESOI at the lower rung; else continue to the cap (onset = SESOI = 0.05). It changes T32 only: the literal rule gave S* = 1200 with D and F both at the floor (validation BSS +0.007 / +0.007; M4 fail), the amended rule gives S* = 4800. Both tuning results are in `tuning_iter4.json`.

## Iteration 4: procedure, tuning findings, deviations

**Procedure.**
- Main seeds 11-20 had never been used. `config_iter4.yaml` was frozen, and its SHA-256 recorded in
  `config_iter4.freeze.json`, before any of them was simulated. `iter4_run.py` refuses to run if the
  file has changed.
- R loads `config_iter3.yaml` verbatim (its SHA-256 is checked at load) with the iteration-3 tuned
  hyperparameters, and is not re-tuned.
- G8, T32 and G32 were tuned on seed 1000 only. A step ladder with the iteration-3 values (lr 1e-3,
  auxiliary weight 1, weight decay 0.1) set S*. An identical grid for D and F at S*
  (lr {1e-3, 3e-3} × auxiliary weight {0.3, 1, 3}, weight decay fixed at 0.1) then chose the
  hyperparameters, and the strengthened M4 was judged on validation against the seed's own M2 ceiling.
- Code is behind config flags. After the iteration-4 changes, iteration 1 (val BSS 0.393417…) and
  iteration 3's co-timed path (0.400705… at 4800 steps) reproduce exactly.

**Generic objective (G8, G32).**
- A random 15% of all encoder tokens (records, traces and signatures alike; independent Bernoulli
  draws) have their content replaced by a learned mask vector; timestamps and token types are kept.
  In D, a masked record token also loses its bound signature.
- Linear heads, one per token type, reconstruct each masked token's original input vector; the loss
  is the MSE averaged over masked tokens.
- The objective runs on a separate encoder-only pass. The attribution loss uses the clean pass, so
  masking never hides D's bound signature from the attribution loss in a way that differs from F.
- The heads and mask vector (3,380 parameters) are identical in D and F, never see the source label,
  do not single out the record-signature pair, and are unused at test.

**Tuning findings (seed 1000; full tables above and in `tuning_iter4.json`).**
1. *G8: F becomes competent only late.* F's validation BSS is 0.04 at 1200 steps and 0.06 at 2400,
   then 0.40 at 4800, while D is 0.35-0.38 throughout. The plateau rule is not met within the cap,
   so S* = 4800. Both select lr 3e-3 and auxiliary weight 3. M4 passes (F at 0.93 of the ceiling).
   F's transition sits close to the step cap, so on some main seeds it may not have happened yet.
2. *T32: neither architecture learns at K = 32 under the targeted objective.* Validation BSS is
   ≤ 0.010 for D and F at 1200, 2400 and 4800 steps and at every grid point; M4 fails. The targeted
   reconstruction is trivially satisfied by D (its record token already carries the bound signature),
   so it puts no pressure on D to find the query's record among 32. That is a hypothesis, not tested.
3. *G32: only D learns.* Under the generic objective at K = 32, D reaches 0.30 at 2400 and 0.37 at
   4800 steps, while F stays ≤ 0.009 at every grid point. M4 fails. With a harder join and a
   join-agnostic objective, the binding is what makes learning possible within the budget.

**Deviations, all decided on the tuning seed before freezing.**
- *Plateau rule amended.* The literal rule fired for T32 at 1200 steps while D and F were both at the
  floor: a pre-learning plateau, not a converged one (G32 shows D taking off only after 1200 steps). The
  amended rule (`iter4_extend.py`, `tuning.plateau_onset`) counts a plateau only once D or F has
  validation BSS ≥ SESOI at the lower rung. It changed T32 only (1200 -> 4800); both results are
  recorded.
- *K = 32 batch.* `batch_windows` went from 32 to 8 (compute and memory). This keeps the number of
  labelled episodes per optimisation step at 256, the same as at K = 8. The 8 n0 training and 2048 test
  episodes are kept.
- *Smaller grid.* The learning-rate grid is {1e-3, 3e-3}; 3e-4 was never selected in iterations 1-3.
  Weight decay is fixed at 0.1, as allowed.
- *F_noz during tuning.* It was trained with the screening hyperparameters; it is a floor and
  insensitive to them.
- *Allocation.* Measured cost per run at 4800 steps was R ~420 s, G8 ~450 s, T32 ~790 s and
  G32 ~975 s (6 parallel workers). R and G8 run on 10 seeds at 1 n0 and 8 n0. T32 and G32 run on
  3 seeds at 8 n0 only. T32 λ and the 1 n0 comparison at K = 32 were dropped. Both K = 32 conditions
  fail M4 on the tuning seed, so their main-seed role is to replicate that pattern on fresh seeds,
  not to estimate a D − F effect against a competent control.

## After the main run (written after inspecting `results_iter4.json`; not preregistered)

1. **K = 8: the equivalence replicates on fresh seeds, and it is not specific to the targeted
   objective.** R gives D − F = +0.001, inside ±0.05. Under the generic, join-agnostic objective (G8) F
   is just as competent, and slightly *better* than D (−0.023, also inside ±0.05). At 1 n0, F is
   slightly better in both R (−0.030) and G8 (−0.020). D has no sample-efficiency advantage at K = 8.

2. **K = 32 shows optimisation barriers, not a binding effect.** Every K = 32 run either jumps out of
   the floor within a few hundred steps (validation BSS 0.01 -> 0.3) or never leaves it. Which
   architecture escapes depends on the objective:
   - under the targeted objective (T32), F escapes on 3/3 seeds and D on 0/3;
   - under the generic objective (G32), D escapes on 3/3 seeds and F on 0/3.

   G32's "support" reading comes with M4 failing, so it repeats the iteration-1 situation (a control
   that cannot learn the join), and T32 shows the reverse. The T32 hyperparameters were chosen among
   grid points that all failed on the tuning seed, so they are essentially arbitrary. Neither K = 32
   contrast is an estimate of what binding buys against a competent control. With 3 seeds and no 1 n0
   runs at K = 32, the honest summary is: at K = 32 and 4800 steps, the learning problem sits at the
   edge of what either architecture can optimise, and the label-free objective decides which one gets
   over the barrier.

3. **The decoder leakage audit fails formally in one seed, and the criterion needs revising.** Seed 13
   (K = 8) has non-trace decoder accuracy 0.3555 against a bound of 1/3 + 0.02 = 0.3533, while its BSS
   is +0.001 (no information). With 2048 test episodes, the accuracy of a calibrated classifier that
   carries no information exceeds that bound in roughly 2-3% of seeds, so a 10-seed audit flags at
   least one seed about a quarter of the time. Pipeline leakage (F_noz) is clean in every condition.
   Recommendation: base the preregistered leakage criterion on BSS, or on an interval for accuracy,
   rather than on raw accuracy.

4. **CPU used: about 6.2 hours**, over the ~5 h target:
   - benchmarks, reproduction checks and smoke tests: ~25 min;
   - tuning: 66 min, plus 34 min for the T32 amendment;
   - main run: 245 min, against an estimated ~170 (the K = 32 runs and the 6-way contention cost more
     than the tuning-seed timings suggested).

   T32 λ and the K = 32 1 n0 runs were dropped beforehand, as recorded in the allocation.

