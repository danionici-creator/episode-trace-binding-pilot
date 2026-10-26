## Iteration 3: strongest admissible control under a shared label-free objective

**Procedure.** All screening and tuning used only the tuning seed (1000). Its information ceiling
was measured with the M2 decoder under the main-run protocol: BSS 0.436 on validation, 0.446 on test.
Candidates a+c, a+b+c and a+b+c+d all carried the co-timed reconstruction objective (weight 1.0 while
screening), applied identically to D, F and F_noz. Each candidate's step budget came from the plateau
rule on a doubling ladder. M4 (F ≥ 0.5 × ceiling and F − F_noz ≥ 0.05) and the selection (highest F
validation BSS among candidates meeting M4) were both judged on validation; test values were never
used. `config_iter3.yaml` was frozen (a+c, 4800 steps) before any main seed was simulated, and the
formal grid of 18 configurations per architecture (lr × weight decay × auxiliary weight
{0.3, 1, 3}, never 0) was then rerun at those settings. Every number is in `candidates_iter3.json`.

1. **Every candidate meets the strengthened M4 by a wide margin.** F reaches 84-86% of the ceiling on
   validation (81-83% on test) and F − F_noz ≈ +0.37. Under the shared objective, the episode-trace
   join barrier of iterations 1-2 disappears whatever the curriculum or the encoder size; the
   objective alone (a+c) suffices.

2. **On the tuning seed F ≈ D for every candidate.** D − F is −0.005, +0.008 and +0.008 on validation
   and −0.023, −0.013 and −0.002 on test, for a+c, a+b+c and a+b+c+d respectively.

3. **Selection: a+c.** F validation BSS is 0.376 at S* = 4800, against 0.370 (a+b+c, S* = 1200) and
   0.367 (a+b+c+d, S* = 1200). The differences between candidates (≤ 0.009) are smaller than the
   sampling error, so the rule picked among practically equal controls. The choice matters for
   compute: a+c needs 4800 steps because F's validation BSS keeps creeping up (+0.004, +0.010, +0.006
   per doubling; the 2400 -> 4800 gain of +0.0101 just misses the 0.01 tolerance), while D peaks at
   2400 (+0.023, then −0.017).

4. **The curriculum overfits when training runs long.** With b, both D and F lose 0.04-0.09 BSS from
   1200 to 2400 steps (iteration 2's a+b+c screen at 2400 is reproduced exactly: D 0.299, F 0.330 on
   test). The plateau rule stops b candidates at 1200 steps.

5. **CPU.** At S* = 4800 a 10-seed learning curve would have pushed the iteration well past the CPU
   target, so the learning curve runs on seeds 1-5; all other conditions use 10 seeds.

6. **Data.** Iteration-3 agent lives are identical to iteration 2's (same role sequence, since the
   largest learning-curve budget is 8 n0 in both), so iterations 2 and 3 are paired seed by seed. The
   tuning-seed life is the one used in iteration 2's candidate search.

7. **Implementation** (all behind config flags; iterations 1 and 2 re-render byte-identical):
   - the auxiliary weight can be tuned through the grid (a hyperparameter overrides the training default);
   - the M2 decoder is also scored on the validation windows (`checks.m2_eval_val`);
   - M4 has a ceiling-share criterion (`checks.m4_ceiling_fraction`);
   - the Sec. 4.7 decision rule (support / falsification with or without equivalence / inconclusive)
     is computed for H1 and for every learning-curve budget, using the (1 − alpha) and (1 − 2 alpha)
     intervals with alpha = 0.05/3.
