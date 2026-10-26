## Iteration 2: making F pass M4, and what the candidate screen shows

**Procedure.** All candidate work used only the tuning seed (1000); `config_iter2.yaml` was frozen
(`selected_candidate: a`, `training.steps: 2400`) before any main seed was simulated. Candidates were
applied identically to D, F and F_noz, with the iteration-1 tuned hyperparameters used only for
screening; the formal 6-configuration tuning grid was then rerun for D and F at the frozen settings.
M4 on the tuning seed was judged on its 2048 held-out test episodes. Every candidate result is
in `candidates_iter2.json` and in the table above.

1. **Candidate (a), longer training, passes M4 by a hair.** On the step ladder F − F_noz is
   +0.037, +0.052, +0.055 and +0.050 at 1200, 2400, 4800 and 9600 steps. Under the preregistered
   plateau rule (best validation BSS gains < 0.01 for both D and F when steps double), S* = 2400,
   decided by F's gain of +0.0095 from 2400 to 4800. M4 at S* = +0.052 against a margin of 0.05.
   Both thresholds are crossed by less than the test-set sampling error (about 0.015), so the
   selection is legitimate under the rule but fragile. D does not benefit from longer training
   (best checkpoint at 800-1100 steps in every rung).

2. **Selected: (a).** It is the first candidate in the preregistered order to pass M4. The only
   change from iteration 1's model and training settings is the step budget (1200 -> 2400).

3. **Post-selection screen (information only, not used for selection).** The remaining candidates
   were screened on the tuning seed at S* = 2400 with screening hyperparameters, one run each:
   - a+b (window-length curriculum K = 1 -> 2 -> 4 -> 8): F − F_noz = +0.051; the curriculum alone does nothing.
   - **a+b+c (+ label-free co-timed reconstruction loss): F − F_noz = +0.333. F reaches test BSS 0.330,
     while D under the same training reaches 0.299.** F's validation curve shows a sharp transition
     at about 1100-1300 steps, after which F uses the trace.
   - a+b+c+d (+ 3-layer, 64-wide, 8-head encoder): F 0.245, D 0.281 (M4 +0.247).
   F's failure in iteration 1 and under (a) is therefore an optimisation barrier, and a label-free
   auxiliary objective removes it. On this seed, with this training, the binding B buys no more
   than that objective does. This is the most important result of iteration 2. It is a single run
   on the tuning seed with untuned screening hyperparameters, and the curriculum and auxiliary loss
   both lowered D (0.35 under a, 0.34 under a+b, 0.30 under a+b+c), so it calls for a confirmatory
   check rather than a conclusion.

4. **Implications for the manuscript.**
   - *M4 as written is too lenient.* "F − F_noz ≥ SESOI" is met by an F that exploits about a tenth of
     the information ceiling, while a reachable F exploits about two thirds (BSS 0.33 against a
     main-seed ceiling of about 0.5). A stronger M4 would
     require F to reach a preregistered fraction of the M2 ceiling, or would select the *strongest*
     admissible candidate on the tuning seed rather than the first that passes.
   - *The H1 contrast needs sharpening.* If D's advantage vanishes against F with a label-free
     co-timed objective, H1 as stated ("explicit binding beats an information-matched control")
     reduces to "binding supplies an inductive bias that a self-supervised objective also supplies".
     The informative contrast is then D vs F + auxiliary objective, possibly as a function of the
     training budget (sample efficiency), and with the auxiliary objective also given to D.
   - *Suggested iteration 3* (the protocol owner's decision; not run here, to keep main-seed data
     uncontaminated by post hoc choices): freeze a+b+c, or a+c, which the fixed order did not test,
     on the tuning seed, rerun the formal tuning grid, and run the same main-seed design.

5. **Implementation.** Curriculum: windows of K = 1, 2, 4, 8 episodes over 15/15/20/50% of the steps,
   with the number of episodes per step held constant and the window placed at a random slot offset
   so that every timestamp is seen. Auxiliary loss (weight 1.0): an MSE between a linear read-out of
   each encoded record token and the signature token with the same episode timestamp. It uses only
   inputs, never the source label. In D the target is the unbound stream signature, so under D_lambda
   it stays correctly paired. The auxiliary head (1,568 parameters) is identical in D and F and is
   unused at test.

6. **Data.** The simulator's window-role sequence depends on the largest learning-curve budget (16 n0
   in iteration 1, 8 n0 here), so iteration-2 agent lives are new draws of the same generative process.
   Iteration 2 is therefore an independent replication of iteration 1 and is not paired with it.
   Iteration 1 still reproduces exactly on its own lives: the tuning-seed D run matches results.json to
   machine precision.
