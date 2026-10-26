# Design notes — iteration 5 (exploratory diagnostic of the K = 32 crossover)

Written on 25 October 2026, before `config_iter5.yaml` was frozen and before any run of this iteration.

## Why this iteration exists

Iteration 4 ran the two conditions with 32 episodes per window on three seeds only. Under the targeted
co-timed objective (T32), D stayed at the floor while F learned; under the generic masked-token
objective (G32), D learned while F stayed at the floor. Two reviews of the preprint (CMU Paper Reviewer
and Google DeepReview, 25 October 2026) independently identified this crossover as the main open issue
and asked whether it is robust and what causes it.

Inspection of the code confirms one candidate mechanism. In `self_models.py`, D adds `s_bind(s_k)` to the
record token before the encoder, and in `train_sm.py` the co-timed objective predicts `s_k` from the
encoded record token (`aux_head(mem[:, :K])`). In D the objective can therefore be met by reading back
its own bound input, an auto-encoding shortcut that gives no pressure to integrate the co-timed trace
tokens. In F the record token does not contain `s_k`, so the same objective forces the join. Whether
this shortcut causes the T32 failure is the hypothesis tested here. A second candidate is that T32's
hyperparameters (learning rate 0.003, auxiliary weight 3.0) were selected on a tuning seed on which
neither architecture learned, i.e. on noise.

## What is changed, and what is not

* `train_sm.py` gains one option, `training.aux_prebind`, absent (false) in every earlier configuration,
  so iterations 1–4 are unchanged. When true, the co-timed target is predicted from a second encoder pass
  in which the bound copy is replaced by zeros. In F, which has no binding, the second pass equals the
  first and F is mathematically unchanged; a smoke test on an unused seed (999) gave identical F scores
  with and without the option.
* No tuning. Every condition uses either the frozen iteration-4 hyperparameters or the iteration-3 tuned
  values (learning rate 0.001, weight decay 0.1, auxiliary weight 1.0), fixed in `config_iter5.yaml`.
* Lives are re-simulated deterministically from their seeds (the iteration-4 caches were not kept); the
  re-simulated ceilings are compared with the recorded ones as part of the audit.

## Conditions (priority order)

| Condition | What it tests |
|---|---|
| T32_pre | frozen T32 settings, co-timed target from the pass with the bound copy zeroed (D only; F unchanged) |
| T32_ext, G32_ext | frozen T32 and G32 settings on seeds 14–20, giving ten seeds per condition |
| T32_R, T32_R_pre | the same two variants with the iteration-3 hyperparameters, separating hyperparameters from the shortcut |
| N32 | no auxiliary objective at K = 32 (D on ten seeds, F on three) |
| T32_long, G32_long | the failing architecture of each condition with 9,600 instead of 4,800 steps (seeds 11–13) |
| R_audit | iteration-4 R on seeds 11–13 re-run from scratch and compared with the recorded scores |

## A priori readings (fixed before running)

* A run has passed the onset of learning if its test BSS is at least 0.05 (SESOI); time to onset is the
  first evaluation step with validation BSS at least 0.05.
* **Shortcut.** If D passes onset in most T32_pre seeds while it fails in T32, the T32 failure is attributed
  to the auto-encoding shortcut. If D fails in both, the shortcut is not a sufficient explanation, and the
  T32_R pair shows whether the hyperparameters selected on the dead tuning seed are responsible.
* **Robustness.** The crossover is called robust if it holds in at least 8 of 10 seeds in each condition.
* **No objective.** N32 shows whether either architecture learns at K = 32 without an auxiliary objective,
  i.e. whether D's success under G32 needs the generic objective at all.
* **Budget.** If a failing architecture learns with 9,600 steps, the failure at 4,800 steps is a censored
  onset, not an inability.
* **Audit.** Reproduction is confirmed if every re-run lies within 0.005 BSS of the recorded value.

Every outcome is reported, whatever its direction. This iteration is exploratory: it diagnoses the
crossover and informs the confirmatory design; it does not test H1.
