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
