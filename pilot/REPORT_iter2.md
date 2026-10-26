# Phase I simulation pilot (iteration 2) — Study 1, *Binding Episodes to Their Internal Effects*

Generated 2026-10-24 23:24:48 by `python pilot/run_pilot.py --config config_iter2.yaml` (56.3 min on 6 CPU workers; Python 3.14.6, torch 2.13.0+cpu). 10 seeds per condition; every seed is a new environment instance and agent life. All numbers are means ± SD across seeds with 95% t-intervals unless stated otherwise. Full numbers: `results_iter2.json`. Preregistered settings: `config_iter2.yaml`.

## Bottom line

- **Feasibility:** all manipulation checks pass. The trace carries source information that the log does not: the M2 ceiling is BSS 0.50 with z vs -0.001 without. The slow state is path-dependent: permuted-order distance is 3.9× the test–retest threshold. No source information reaches the self-models except through the trace.
- **M4 (control competence):** F − F_noz = +0.057 [+0.048, +0.066] BSS; at least SESOI in 7/10 seeds; PASS. F exploits 11% of the information ceiling.
- **H1:** D exploits 86% of the information ceiling, F 11%. D − F = +0.373 BSS [+0.357, +0.389], which remains above SESOI; the whole 95% CI lies above it. The SD of the paired difference over 10 seeds is 0.022.
- **H2:** D's BSS declines monotonically with training-time misalignment (slope -0.380 per unit λ) and reaches D₁ − F₀ = -0.007 [-0.010, -0.004] at λ = 1, the pattern H2 predicts; D with correct training but misbound test signatures falls far below F (-0.544).
- **Post-selection screen (tuning seed only, information only):** candidate a+b+c raises F − F_noz from +0.052 (selected a) to +0.333. On that seed F reaches +0.330 against D's +0.299 under the same training, so D − F = -0.031. The selected F passes M4 but is far from the strongest admissible control, and D's advantage over it says little about D against a competent F. M4 as preregistered is too lenient (design notes, points 3–4). This is a single untuned run, not a main-seed result.

## Instantiation in one paragraph

A persistent environment (world map o = tanh(Ac), 4 content kinds, a scripted other agent) produces fixed-length episodes (6 steps) with a latent label y_k in {own, imported, inferred}, drawn i.i.d. and independent of content kind. **Own**: repeated action–outcome trials with sensory and outcome noise. **Imported**: the other agent's message (cue through a 12-d shared code; outcome from the sender's own, imperfect world model). **Inferred**: rehearsal that recombines the item with an earlier retained cue, its target partly regenerated from the slow memory U. Every route drives the same delta-rule update of the slow state h = (W, U) (fast synapse consolidating into a slow variable), with the same gain. M: z_t = R vec(h_t − h_{t−1})/η + N(0, 0.25²), R a fixed Gaussian projection to 16 dimensions; Φ: s_k = (mean, SD) of z over the episode. Log records and queries are route-independent random descriptors (no provenance flag exists to strip). Self-models see windows of 8 episodes: record tokens, per-step trace tokens (z_t, timestamp, boundary flag) and an unbound signature stream (s_k, stamped with its episode's timestamp). **F** is a 2+2-layer memory-augmented transformer over these streams; **D** is the same network plus the binding B (s_k written into the record token of e_k). Active parameters: D 82,323, F 82,291 (F's feed-forward width 100 vs 96; mismatch -0.04%). Identical steps (2400), batch, schedule, data order and tuning budget (6 configurations each; selected D {'lr': 0.001, 'weight_decay': 0.0}, F {'lr': 0.003, 'weight_decay': 0.1}). Confirmatory budget 8 n0 = 8,192 labelled training episodes; 2,048 novel test episodes per run.

## Manipulation checks and leakage audit

| Check | Criterion | Result | Verdict |
|---|---|---|---|
| M1 path dependence | median d(h_σ, h_id) > δ_C (95th pct. test–retest) | distance ratio +3.948 ± 0.290 (min 3.40) | PASS (10/10 seeds) |
| M2 information ceiling | BSS(decoder with z) − BSS(without z) ≥ 0.05 | Δ = +0.499 ± 0.021 (min +0.468) | PASS (10/10 seeds) |
| Leakage, decoder | non-trace inputs: BSS ≤ 0.02 and accuracy ≤ 1/3 + 0.02 | BSS -0.001 ± 0.001, accuracy 0.334 (max 0.345) | PASS |
| Leakage, pipeline | F trained and tested without z and s: BSS ≤ 0.02 | BSS -0.001 ± 0.001 | PASS |
| M3 capacity match | active parameters within ±2% | -0.04% | PASS |
| M4 control competence | BSS(F) − BSS(F_noz) ≥ 0.05 | +0.057 ± 0.013 [+0.048, +0.066] | PASS (7/10 seeds individually) |

### M4 candidates tried on the tuning seed (seed 1000 only; preregistered order a → a+b → a+b+c → a+b+c+d)

Iteration-1 reproduction check (D on the tuning seed with config.yaml): val BSS 0.3934 vs 0.3934 in results.json (identical).

| Candidate | Steps | D val / test | F val / test | F_noz test | M4 = F − F_noz (test) | M4 |
|---|---|---|---|---|---|---|
| a (ladder) | 1200 | +0.376 / +0.360 | +0.043 / +0.036 | -0.001 | +0.037 | fail |
| a (plateau budget) | 2400 | +0.374 / +0.352 | +0.054 / +0.050 | -0.002 | +0.052 | pass |
| a (ladder) | 4800 | +0.363 / +0.360 | +0.064 / +0.053 | -0.002 | +0.055 | pass |
| a (ladder) | 9600 | +0.359 / +0.354 | +0.068 / +0.048 | -0.002 | +0.050 | pass |
| a+b — screened after selection | 2400 | +0.337 / +0.336 | +0.070 / +0.048 | -0.004 | +0.051 | pass |
| a+b+c — screened after selection | 2400 | +0.327 / +0.299 | +0.327 / +0.330 | -0.003 | +0.333 | pass |
| a+b+c+d — screened after selection | 2400 | +0.320 / +0.281 | +0.276 / +0.245 | -0.002 | +0.247 | pass |

Candidate a: plateau budget S* = 2400 steps (plateau reached; rule: best validation BSS improves by < 0.01 for both D and F when the steps are doubled). M4 for candidate a is judged at S*.

**Selected:** a, the first candidate in the preregistered order to pass M4 on the tuning seed.

Information ceiling (independent decoder, correct pairing, 81,920 training episodes = 10 × the confirmatory budget), test BSS:

| Decoder features | BSS |
|---|---|
| z for every step + s_k + record + timestamp (ceiling with z) | +0.498 ± 0.022 |
| s_k + record + timestamp (what Φ retains) | +0.484 ± 0.022 |
| record + query + timestamp + window log context (ceiling without z) | -0.001 ± 0.001 |
| control task: full features, shuffled training labels | -0.001 ± 0.002 |
| full features with z and s_k taken from another episode of the window | -0.001 ± 0.001 |

D exploits 86% of the ceiling, F 11%.

## Primary outcome: D vs F (H1)

Novel test episodes, confirmatory budget 8 n0.

| Condition | BSS | log loss | accuracy | ECE |
|---|---|---|---|---|
| D (explicit binding) | +0.429 ± 0.029 [+0.408, +0.450] | 0.650 | 0.722 | 0.043 |
| F (information control) | +0.056 ± 0.012 [+0.048, +0.065] | 1.041 | 0.456 | 0.020 |
| F without z and s (floor) | -0.001 ± 0.001 [-0.001, -0.000] | 1.099 | 0.333 | 0.014 |

**Paired D − F (BSS):** mean +0.373, **SD of the paired difference 0.022** (80% upper confidence limit 0.029), 95% CI [+0.357, +0.389], bootstrap 95% CI [+0.359, +0.385], (1 − 2α) CI with α = 0.05/3 [+0.355, +0.391]; d_z = 16.8. Test-set sampling SE within a seed 0.016; implied between-run SD 0.015.

Secondary outcomes, paired D − F: log loss -0.391 [-0.408, -0.375], accuracy +0.265 [+0.258, +0.273], ECE +0.023 [+0.014, +0.032].

## Misalignment λ (H2)

D_λ: a fraction λ of the bound signatures is permuted across episodes of the same window during training (unbound streams intact). F_λ: the same permutation applied to F's trace stream and signature stream. Testing always uses correct pairing.

| λ | D_λ BSS | F_λ BSS | D_λ − F₀ | D_λ − F_λ |
|---|---|---|---|---|
| 0 | +0.429 ± 0.029 | +0.056 ± 0.012 | +0.373 [+0.357, +0.389] | +0.373 [+0.357, +0.389] |
| 0.5 | +0.300 ± 0.105 | +0.027 ± 0.004 | +0.243 [+0.164, +0.323] | +0.273 [+0.196, +0.349] |
| 1 | +0.050 ± 0.010 | -0.001 ± 0.001 | -0.007 [-0.010, -0.004] | +0.050 [+0.043, +0.058] |

Per-seed slope of BSS on λ: D -0.380 [-0.396, -0.364], F -0.057 [-0.066, -0.048]. D₁ − F₀ = -0.007 [-0.010, -0.004]; D₁ retains -2% of the D − F advantage (H2 requires < 50%). D means decrease monotonically in λ: yes.

D_λ^test (correct training, all test signatures permuted): BSS -0.487 ± 0.031; minus F -0.544 [-0.560, -0.528] (expected below F).

## Learning curve (D₀, F₀; 5 diagnostic seeds)

| budget | training episodes | D BSS | F BSS | D − F |
|---|---|---|---|---|
| 1 n0 | 1,024 | +0.275 ± 0.016 | +0.029 ± 0.007 | +0.246 [+0.224, +0.268] |
| 4 n0 | 4,096 | +0.390 ± 0.030 | +0.051 ± 0.014 | +0.338 [+0.309, +0.368] |
| 8 n0 | 8,192 | +0.432 ± 0.032 | +0.059 ± 0.014 | +0.373 [+0.346, +0.400] |

## Inputs for the power analysis

With the observed SD of the paired difference (0.022 BSS), two-sided α = 0.05/3 and power 0.90, detecting an effect of size SESOI = 0.05 needs ((z₁₋α/₂ + z₀.₉₀) · SD / SESOI)² = **3 paired run(s)** (5 with the 80% upper confidence limit of SD). The manuscript's illustration assumed SD = 0.08 (≈ 35 runs). For H2, the per-seed λ slope of D has SD 0.022 BSS per unit λ; the D₁ − F₀ contrast has SD 0.005. The observed D − F effect (+0.373) is more than three times SESOI, so in this instantiation H1 does not drive the run count. The planned 40 runs are set by the equivalence and λ contrasts and by H5, and above all by whether a stronger F changes the effect size.

## Verdicts and automatically detected problems

- **H1 (pilot reading, not confirmatory):** D − F = +0.373 BSS, (1 − 2α) CI [+0.355, +0.391]; the interval excludes 0 and the estimate exceeds SESOI.
- **H2 (pilot reading):** D slope -0.380 per unit λ; D₁ retains -2% of the advantage.
- ⚠ Intermediate misalignment is unstable: the between-seed SD of D_λ reaches 0.105 vs 0.029 at λ = 0. Some seeds discover the partly wrong binding late in training and others not at all. The λ slope therefore mixes reliance on the binding with learning speed, and the intermediate λ points dominate the variance of the H2 test.

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

