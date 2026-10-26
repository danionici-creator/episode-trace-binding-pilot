# Phase I simulation pilot — Study 1, *Binding Episodes to Their Internal Effects*

Generated 2026-10-24 21:36:31 by `python pilot/run_pilot.py` (31.9 min on 6 CPU workers; Python 3.14.6, torch 2.13.0+cpu). 10 seeds per condition; every seed is a new environment instance and agent life. All numbers are means ± SD across seeds with 95% t-intervals unless stated otherwise. Full numbers: `results.json`. Preregistered settings: `config.yaml`.

## Bottom line

- **Feasibility:** all manipulation checks pass. The trace carries source information that the log does not: the M2 ceiling is BSS 0.50 with z vs -0.001 without. The slow state is path-dependent: permuted-order distance is 3.9× the test–retest threshold. No source information reaches the self-models except through the trace.
- **H1:** D exploits 86% of the information ceiling, F 9%. D − F = +0.384 BSS [+0.373, +0.395]; the SD of the paired difference over 10 seeds is 0.015.
- **H2:** D's BSS declines monotonically with training-time misalignment (slope -0.398 per unit λ) and reaches D₁ − F₀ = +0.001 [-0.001, +0.004] at λ = 1, the pattern H2 predicts; D with correct training but misbound test signatures falls far below F (-0.516).
- **Variance:** the across-seed SD of D − F (0.015) is accounted for by test-set sampling (within-seed SE 0.015 with 2,048 test episodes per run). Between-run variance is negligible here, so the confirmatory SD is set mainly by the number of test episodes per run, not by the number of runs.
- **Main caveat:** F barely uses the trace. F − F_noz = +0.048, below SESOI; with 4× the steps F reaches only +0.065; over a 16-fold range of training data F moves only from +0.026 to +0.046, while D moves from +0.248 to +0.441. The effect therefore measures the two-hop episode–trace join that a generic transformer does not discover within budget. That is the inductive-bias claim in its strongest form, but it also means the H1 effect size, and with it the power analysis, says nothing yet about a *competent* control. A control-competence check and a stronger generic F are needed before confirmatory data collection (design note 8).

## Instantiation in one paragraph

A persistent environment (world map o = tanh(Ac), 4 content kinds, a scripted other agent) produces fixed-length episodes (6 steps) with a latent label y_k in {own, imported, inferred}, drawn i.i.d. and independent of content kind. **Own**: repeated action–outcome trials with sensory and outcome noise. **Imported**: the other agent's message (cue through a 12-d shared code; outcome from the sender's own, imperfect world model). **Inferred**: rehearsal that recombines the item with an earlier retained cue, its target partly regenerated from the slow memory U. Every route drives the same delta-rule update of the slow state h = (W, U) (fast synapse consolidating into a slow variable), with the same gain. M: z_t = R vec(h_t − h_{t−1})/η + N(0, 0.25²), R a fixed Gaussian projection to 16 dimensions; Φ: s_k = (mean, SD) of z over the episode. Log records and queries are route-independent random descriptors (no provenance flag exists to strip). Self-models see windows of 8 episodes: record tokens, per-step trace tokens (z_t, timestamp, boundary flag) and an unbound signature stream (s_k, stamped with its episode's timestamp). **F** is a 2+2-layer memory-augmented transformer over these streams; **D** is the same network plus the binding B (s_k written into the record token of e_k). Active parameters: D 82,323, F 82,291 (F's feed-forward width 100 vs 96; mismatch -0.04%). Identical steps (1200), batch, schedule, data order and tuning budget (6 configurations each; selected D {'lr': 0.003, 'weight_decay': 0.0}, F {'lr': 0.003, 'weight_decay': 0.1}). Confirmatory budget 8 n0 = 8,192 labelled training episodes; 2,048 novel test episodes per run.

## Manipulation checks and leakage audit

| Check | Criterion | Result | Verdict |
|---|---|---|---|
| M1 path dependence | median d(h_σ, h_id) > δ_C (95th pct. test–retest) | distance ratio +3.948 ± 0.290 (min 3.40) | PASS (10/10 seeds) |
| M2 information ceiling | BSS(decoder with z) − BSS(without z) ≥ 0.05 | Δ = +0.499 ± 0.018 (min +0.468) | PASS (10/10 seeds) |
| Leakage, decoder | non-trace inputs: BSS ≤ 0.02 and accuracy ≤ 1/3 + 0.02 | BSS -0.001 ± 0.001, accuracy 0.332 (max 0.351) | PASS |
| Leakage, pipeline | F trained and tested without z and s: BSS ≤ 0.02 | BSS -0.001 ± 0.001 | PASS |
| M3 capacity match | active parameters within ±2% | -0.04% | PASS |

Information ceiling (independent decoder, correct pairing, 81,920 training episodes = 10 × the confirmatory budget), test BSS:

| Decoder features | BSS |
|---|---|
| z for every step + s_k + record + timestamp (ceiling with z) | +0.498 ± 0.018 |
| s_k + record + timestamp (what Φ retains) | +0.485 ± 0.015 |
| record + query + timestamp + window log context (ceiling without z) | -0.001 ± 0.001 |
| control task: full features, shuffled training labels | -0.001 ± 0.003 |
| full features with z and s_k taken from another episode of the window | -0.001 ± 0.001 |

D exploits 86% of the ceiling, F 9%.

## Primary outcome: D vs F (H1)

Novel test episodes, confirmatory budget 8 n0.

| Condition | BSS | log loss | accuracy | ECE |
|---|---|---|---|---|
| D (explicit binding) | +0.431 ± 0.014 [+0.421, +0.441] | 0.647 | 0.720 | 0.036 |
| F (information control) | +0.047 ± 0.008 [+0.041, +0.052] | 1.051 | 0.441 | 0.017 |
| F_long (4× steps, diagnostic) | +0.065 ± 0.009 [+0.054, +0.075] | 1.033 | 0.468 | 0.024 |
| F without z and s (floor) | -0.001 ± 0.001 [-0.002, -0.001] | 1.100 | 0.330 | 0.021 |

**Paired D − F (BSS):** mean +0.384, **SD of the paired difference 0.015** (80% upper confidence limit 0.020), 95% CI [+0.373, +0.395], bootstrap 95% CI [+0.375, +0.393], (1 − 2α) CI with α = 0.05/3 [+0.372, +0.396]; d_z = 25.4. Test-set sampling SE within a seed 0.015; implied between-run SD 0.000 (the across-seed SD is accounted for by test-set sampling).

Secondary outcomes, paired D − F: log loss -0.404 [-0.417, -0.390], accuracy +0.279 [+0.269, +0.288], ECE +0.019 [+0.013, +0.026].

F with 4× the optimisation steps (5 diagnostic seeds): BSS +0.065 ± 0.009; F_long − F +0.019 [+0.008, +0.030]; D − F_long +0.364 [+0.343, +0.385].

## Misalignment λ (H2)

D_λ: a fraction λ of the bound signatures is permuted across episodes of the same window during training (unbound streams intact). F_λ: the same permutation applied to F's trace stream and signature stream. Testing always uses correct pairing.

| λ | D_λ BSS | F_λ BSS | D_λ − F₀ | D_λ − F_λ |
|---|---|---|---|---|
| 0 | +0.431 ± 0.014 | +0.047 ± 0.008 | +0.384 [+0.373, +0.395] | +0.384 [+0.373, +0.395] |
| 0.25 | +0.400 ± 0.026 | +0.038 ± 0.005 | +0.353 [+0.333, +0.373] | +0.362 [+0.343, +0.381] |
| 0.5 | +0.293 ± 0.094 | +0.028 ± 0.004 | +0.246 [+0.180, +0.313] | +0.265 [+0.198, +0.331] |
| 1 | +0.048 ± 0.007 | -0.001 ± 0.001 | +0.001 [-0.001, +0.004] | +0.050 [+0.045, +0.054] |

Per-seed slope of BSS on λ: D -0.398 [-0.414, -0.383], F -0.049 [-0.054, -0.044]. D₁ − F₀ = +0.001 [-0.001, +0.004]; D₁ retains 0% of the D − F advantage (H2 requires < 50%). D means decrease monotonically in λ: yes.

D_λ^test (correct training, all test signatures permuted): BSS -0.469 ± 0.026; minus F -0.516 [-0.536, -0.496] (expected below F).

## Learning curve (D₀, F₀; 5 diagnostic seeds)

| budget | training episodes | D BSS | F BSS | D − F |
|---|---|---|---|---|
| 1 n0 | 1,024 | +0.248 ± 0.013 | +0.026 ± 0.007 | +0.222 [+0.203, +0.240] |
| 4 n0 | 4,096 | +0.380 ± 0.019 | +0.042 ± 0.010 | +0.338 [+0.314, +0.362] |
| 8 n0 | 8,192 | +0.428 ± 0.015 | +0.045 ± 0.009 | +0.383 [+0.361, +0.405] |
| 16 n0 | 16,384 | +0.441 ± 0.023 | +0.046 ± 0.010 | +0.396 [+0.371, +0.420] |

## Inputs for the power analysis

With the observed SD of the paired difference (0.015 BSS), two-sided α = 0.05/3 and power 0.90, detecting an effect of size SESOI = 0.05 needs ((z₁₋α/₂ + z₀.₉₀) · SD / SESOI)² = **2 paired run(s)** (3 with the 80% upper confidence limit of SD). The manuscript's illustration assumed SD = 0.08 (≈ 35 runs). For H2, the per-seed λ slope of D has SD 0.022 BSS per unit λ; the D₁ − F₀ contrast has SD 0.004. The observed D − F effect (+0.384) is more than three times SESOI, so in this instantiation H1 does not drive the run count. The planned 40 runs are set by the equivalence and λ contrasts and by H5, and above all by whether a stronger F changes the effect size.

## Verdicts and automatically detected problems

- **H1 (pilot reading, not confirmatory):** D − F = +0.384 BSS, (1 − 2α) CI [+0.372, +0.396]; the interval excludes 0 and the estimate exceeds SESOI.
- **H2 (pilot reading):** D slope -0.398 per unit λ; D₁ retains 0% of the advantage.
- ⚠ F is within SESOI of the no-trace floor (F − F_noz = +0.048). The control uses the trace only marginally, so D − F measures the episode–trace join barrier rather than a graded advantage over a competent control. A stronger generic F, or many more steps or data, is needed before the H1 contrast can be called a fair test (design notes 5 and 8).
- ⚠ Intermediate misalignment is unstable: the between-seed SD of D_λ reaches 0.094 vs 0.014 at λ = 0. Some seeds discover the partly wrong binding late in training and others not at all. The λ slope therefore mixes reliance on the binding with learning speed, and the intermediate λ points dominate the variance of the H2 test.
- ⚠ D is still improving at the end of training: its best checkpoint is in the last two evaluations in 9/10 seeds. The step budget truncates D (conservative for H1) and makes the intermediate λ points sensitive to it.

## Design problems found while instantiating the protocol

All iterations below used only the tuning seed (1000). No main-seed data were generated or
inspected before the configuration in `config.yaml` was frozen.

1. **The M2 ceiling saturates unless the measurement is noisy.** With sigma_z = 0.05 the
   independent decoder attributed source almost perfectly (BSS 0.87), leaving no room for a
   D-F difference. sigma_z was raised to 0.25 (ceiling ~0.44). The ceiling is a steep function
   of measurement noise (0.44 -> 0.08 -> 0.00 for sigma_z = 0.25, 0.5, 1.0). The manuscript should
   state that M and its noise level are calibrated on pilot data so that M2 is neither failed
   nor saturated, and that this calibration uses the decoder only, never D or F.

2. **Record identity is a memorisation shortcut.** When each episode record is a unique
   descriptor, both architectures memorise record -> label on the training set (training loss -> 0,
   validation BSS < 0). The record descriptors carry no source information by design, so they are
   redrawn at every presentation of a training window (`training.redraw_descriptors`). Constraint
   (iii) of Sec. 4.3 therefore needs a training-side counterpart: the record must not be usable as
   a lookup key for the label.

3. **Whole-window memorisation.** After (2), F still memorised (window trace fingerprint, content
   kind) -> label. Training windows are therefore re-assembled from random training episodes at
   every presentation (`training.recompose_windows`). Each episode keeps its own record, trace
   segment and label, so the experience budget (8 n0 labelled episodes) is unchanged; test windows
   remain contiguous stretches of the agent's life.

4. **Phi confound in the D-F contrast (important for the manuscript).** As written in Table 2, D
   receives s_k = Phi(z) bound to the record, while F receives only the raw per-step stream z. D-F
   then confounds the binding with the preregistered summary Phi (F would have to learn the
   within-episode mean and SD itself). In this pilot both D and F receive s_k as an *unbound*
   signature stream (one token per episode, stamped with the same timestamp as its record), and D
   additionally receives B. D and F then differ only in whether the association is given as
   structure. Recommended change to Table 2: "F: episodes, log, trace stream z, signature stream
   s, timestamps, boundaries".

5. **F does not discover the episode-trace join.** On the tuning seed, under every generic variant
   tried, F stayed near the no-trace floor (test BSS 0.00-0.05) while D reached 0.36-0.38 of a
   ~0.44 ceiling: plain timestamps; episode-level timestamps shared by both streams (an equality
   join); an ALiBi-style temporal-locality attention bias; an unbound signature stream; and 4x the
   optimisation steps (6000 steps: BSS 0.037, still rising slowly). The join is a two-hop
   retrieval (query -> record by content -> trace by timestamp). Neither hop is rewarded until the
   other works, so the model stalls on a plateau, and D skips the second hop. The D-F
   difference in this instantiation therefore mostly measures this optimisation barrier. That is
   the inductive-bias claim of the manuscript, but reviewers will call it true by construction. The
   main run includes F_long (4x steps) and a learning curve so that the size of the barrier is
   reported rather than assumed.

6. **Speed.** Dropout on the attention matrices accounted for ~35% of CPU time; dropout was set to
   0 for both architectures (weight decay remains in the tuning grid). Windows were reduced from 16
   to 8 episodes (n0 doubled to 128 windows, so the n0 episode count is unchanged).

7. **Validation under misalignment.** For D_lambda and F_lambda the validation windows used for
   checkpoint selection carry the same training-time misalignment, so model selection never sees
   correctly paired data that the condition is supposed to lack. Testing always uses correct pairing.

8. **Recommended addition to the manuscript: a control-competence check.** M2 guarantees that the
   trace carries source information; nothing in the protocol guarantees that F can use it. If F stays
   at the no-trace floor (as here), D - F is large but tells a reviewer only that F failed, and the
   effect size is useless for powering a comparison with a competent control. Suggested criterion,
   to be checked like M1-M3 before the confirmatory test: F must exceed F_noz by at least SESOI at the
   confirmatory budget (or reach a preregistered fraction of the M2 ceiling). If it does not, the
   F architecture or its budget is revised on pilot data before confirmatory collection. Candidate
   stronger generic controls: more encoder depth or heads, a curriculum that starts with short
   windows (K = 1-2, where the join is trivial) and grows K, auxiliary self-supervised objectives
   that reward attending across co-timed tokens, or a recurrent memory that writes the episode summary
   at each boundary (closest to D, and a reviewer may call it binding by another name).

9. **Not piloted here.** H5 (transfer to self-prediction of performance change), counterfactual
   episodes (Sec. 4.4) and the hierarchical mixed model are not part of this pilot; paired
   seed-level contrasts are reported instead, and per-episode Brier scores are saved
   (`cache/per_episode_brier.npz`) for a later mixed-model or simulation-based power analysis.

