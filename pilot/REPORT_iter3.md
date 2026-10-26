# Phase I simulation pilot (iteration 3) — Study 1, *Binding Episodes to Their Internal Effects*

Generated 2026-10-25 02:51:57 by `python pilot/run_pilot.py --config config_iter3.yaml` (159.7 min on 6 CPU workers; Python 3.14.6, torch 2.13.0+cpu). 10 seeds per condition; every seed is a new environment instance and agent life. All numbers are means ± SD across seeds with 95% t-intervals unless stated otherwise. Full numbers: `results_iter3.json`. Preregistered settings: `config_iter3.yaml`.

## Bottom line

- **Feasibility:** all manipulation checks pass. The trace carries source information that the log does not: the M2 ceiling is BSS 0.50 with z vs -0.001 without. The slow state is path-dependent: permuted-order distance is 3.9× the test–retest threshold. No source information reaches the self-models except through the trace.
- **M4 (control competence, strengthened):** F reaches 0.87 [+0.849, +0.889] of the information ceiling (required ≥ 0.5; 10/10 seeds) and F − F_noz = +0.434 [+0.415, +0.453] BSS (required ≥ 0.05; 10/10 seeds); both criteria are met in 10/10 seeds; **PASS**.
- **H1:** D exploits 87% of the information ceiling, F 87%. D − F = +0.002 BSS [-0.008, +0.012], which falls below SESOI. The SD of the paired difference over 10 seeds is 0.014.
- **Decision rule (Sec. 4.7, pilot reading):** D − F = +0.002; (1 − α) CI [-0.011, +0.015], (1 − 2α) CI [-0.009, +0.013] with α = 0.05/3, equivalence bounds ±0.05. **Falsification (equivalent to zero within ±SESOI)**: the (1 − 2α) interval lies entirely below +SESOI and inside ±SESOI: D and F are equivalent within the SESOI.
- **Learning curve (5 seeds), D − F by training budget:** 1 n0: +0.010 [-0.081, +0.101] (inconclusive); 2 n0: +0.028 [-0.017, +0.073] (inconclusive); 4 n0: +0.009 [-0.008, +0.025] (falsification); 8 n0: -0.004 [-0.007, -0.001] (falsification). D's advantage does not reach SESOI at any budget.
- **H2:** with no D − F advantage of SESOI at λ = 0 there is little for misalignment to remove, so H2 is not informative here. D's own BSS falls from +0.435 to +0.413 as λ goes from 0 to 1, and F's from +0.433 to -0.001; D₁ − F₀ = -0.020 [-0.032, -0.008]. D barely depends on the bound signatures. Trained with all of them misbound, it learns the pairing from the unbound streams instead, while F, whose only pairing is the permuted stream, falls to the floor. Yet with correct training, misbound test signatures drive D to -0.500: D uses B when it is present but does not need it.

## Instantiation in one paragraph

A persistent environment (world map o = tanh(Ac), 4 content kinds, a scripted other agent) produces fixed-length episodes (6 steps) with a latent label y_k in {own, imported, inferred}, drawn i.i.d. and independent of content kind. **Own**: repeated action–outcome trials with sensory and outcome noise. **Imported**: the other agent's message (cue through a 12-d shared code; outcome from the sender's own, imperfect world model). **Inferred**: rehearsal that recombines the item with an earlier retained cue, its target partly regenerated from the slow memory U. Every route drives the same delta-rule update of the slow state h = (W, U) (fast synapse consolidating into a slow variable), with the same gain. M: z_t = R vec(h_t − h_{t−1})/η + N(0, 0.25²), R a fixed Gaussian projection to 16 dimensions; Φ: s_k = (mean, SD) of z over the episode. Log records and queries are route-independent random descriptors (no provenance flag exists to strip). Self-models see windows of 8 episodes: record tokens, per-step trace tokens (z_t, timestamp, boundary flag) and an unbound signature stream (s_k, stamped with its episode's timestamp). **F** is a 2+2-layer memory-augmented transformer over these streams; **D** is the same network plus the binding B (s_k written into the record token of e_k). Active parameters: D 83,891, F 83,859 (F's feed-forward width 100 vs 96; mismatch -0.04%). Both carry a label-free auxiliary loss: each encoded record token reconstructs its co-timed signature (weight tuned per architecture from [0.3, 1.0, 3.0]; selected D 1.0, F 1.0). Identical steps (4800), batch, schedule, data order and tuning budget (18 configurations each; selected D {'lr': 0.001, 'weight_decay': 0.1, 'aux_cotimed_weight': 1.0}, F {'lr': 0.001, 'weight_decay': 0.1, 'aux_cotimed_weight': 1.0}). Confirmatory budget 8 n0 = 8,192 labelled training episodes; 2,048 novel test episodes per run.

## Manipulation checks and leakage audit

| Check | Criterion | Result | Verdict |
|---|---|---|---|
| M1 path dependence | median d(h_σ, h_id) > δ_C (95th pct. test–retest) | distance ratio +3.948 ± 0.290 (min 3.40) | PASS (10/10 seeds) |
| M2 information ceiling | BSS(decoder with z) − BSS(without z) ≥ 0.05 | Δ = +0.499 ± 0.021 (min +0.468) | PASS (10/10 seeds) |
| Leakage, decoder | non-trace inputs: BSS ≤ 0.02 and accuracy ≤ 1/3 + 0.02 | BSS -0.001 ± 0.001, accuracy 0.334 (max 0.345) | PASS |
| Leakage, pipeline | F trained and tested without z and s: BSS ≤ 0.02 | BSS -0.001 ± 0.001 | PASS |
| M3 capacity match | active parameters within ±2% | -0.04% | PASS |
| M4 control competence | BSS(F) ≥ 0.5 × ceiling and BSS(F) − BSS(F_noz) ≥ 0.05 | share +0.869 ± 0.028; F − F_noz +0.434 ± 0.026 [+0.415, +0.453] | PASS (10/10 seeds meet both individually) |

### Candidates screened on the tuning seed (seed 1000 only; all with the co-timed objective)

Information ceiling on the tuning seed (M2 decoder, same protocol): validation BSS 0.436, test 0.446. M4 = F ≥ 0.5 × ceiling and F − F_noz ≥ 0.05, judged on validation (where selection happens); test values are reported only.

| Candidate | S* (steps) | D val / test | F val / test | F_noz val / test | F share of ceiling, val / test | F − F_noz, val / test | M4 (val) |
|---|---|---|---|---|---|---|---|
| a+c **(selected)** | 4800 | +0.371 / +0.340 | +0.376 / +0.363 | +0.001 / -0.002 | 0.86 / 0.81 | +0.375 / +0.365 | pass |
| a+b+c | 1200 | +0.378 / +0.356 | +0.370 / +0.369 | -0.000 / -0.001 | 0.85 / 0.83 | +0.370 / +0.370 | pass |
| a+b+c+d | 1200 | +0.375 / +0.366 | +0.367 / +0.368 | -0.000 / -0.001 | 0.84 / 0.82 | +0.368 / +0.369 | pass |

Step ladders (best validation BSS; S* = smallest budget after which doubling gains < 0.01 for both D and F):

| Candidate | 1200 steps: D / F | 2400 steps: D / F | 4800 steps: D / F | 9600 steps: D / F |
|---|---|---|---|---|
| a+c | +0.365 / +0.362 | +0.388 / +0.366 | +0.371 / +0.376 | +0.372 / +0.382 |
| a+b+c | +0.378 / +0.370 | +0.327 / +0.327 |
| a+b+c+d | +0.375 / +0.367 | +0.320 / +0.276 |

**Selected:** a+c at 4800 steps. Rule: highest F validation BSS among candidates meeting M4.

Information ceiling (independent decoder, correct pairing, 81,920 training episodes = 10 × the confirmatory budget), test BSS:

| Decoder features | BSS |
|---|---|
| z for every step + s_k + record + timestamp (ceiling with z) | +0.498 ± 0.022 |
| s_k + record + timestamp (what Φ retains) | +0.484 ± 0.022 |
| record + query + timestamp + window log context (ceiling without z) | -0.001 ± 0.001 |
| control task: full features, shuffled training labels | -0.001 ± 0.002 |
| full features with z and s_k taken from another episode of the window | -0.001 ± 0.001 |

D exploits 87% of the ceiling, F 87%.

## Primary outcome: D vs F (H1)

Novel test episodes, confirmatory budget 8 n0.

| Condition | BSS | log loss | accuracy | ECE |
|---|---|---|---|---|
| D (explicit binding) | +0.435 ± 0.030 [+0.414, +0.456] | 0.644 | 0.726 | 0.050 |
| F (information control) | +0.433 ± 0.026 [+0.414, +0.452] | 0.649 | 0.726 | 0.051 |
| F without z and s (floor) | -0.001 ± 0.001 [-0.001, -0.000] | 1.099 | 0.330 | 0.019 |

**Paired D − F (BSS):** mean +0.002, **SD of the paired difference 0.014** (80% upper confidence limit 0.018), 95% CI [-0.008, +0.012], bootstrap 95% CI [-0.005, +0.011], (1 − 2α) CI with α = 0.05/3 [-0.009, +0.013], (1 − α) CI [-0.011, +0.015]; d_z = 0.2. Test-set sampling SE within a seed 0.010; implied between-run SD 0.009.

Secondary outcomes, paired D − F: log loss -0.006 [-0.017, +0.005], accuracy -0.000 [-0.009, +0.008], ECE -0.000 [-0.008, +0.007].

## Misalignment λ (H2)

D_λ: a fraction λ of the bound signatures is permuted across episodes of the same window during training (unbound streams intact). F_λ: the same permutation applied to F's trace stream and signature stream. Testing always uses correct pairing.

| λ | D_λ BSS | F_λ BSS | D_λ − F₀ | D_λ − F_λ |
|---|---|---|---|---|
| 0 | +0.435 ± 0.030 | +0.433 ± 0.026 | +0.002 [-0.008, +0.012] | +0.002 [-0.008, +0.012] |
| 0.5 | +0.422 ± 0.032 | +0.041 ± 0.012 | -0.011 [-0.025, +0.003] | +0.381 [+0.365, +0.398] |
| 1 | +0.413 ± 0.031 | -0.001 ± 0.002 | -0.020 [-0.032, -0.008] | +0.413 [+0.391, +0.436] |

Per-seed slope of BSS on λ: D -0.022 [-0.033, -0.011], F -0.434 [-0.453, -0.414]. D₁ − F₀ = -0.020 [-0.032, -0.008]; the retained fraction of the D − F advantage is not meaningful, as there is no advantage at λ = 0. D means decrease monotonically in λ: yes.

D_λ^test (correct training, all test signatures permuted): BSS -0.500 ± 0.031; minus F -0.933 [-0.963, -0.904] (expected below F).

## Learning curve (D₀, F₀; 5 diagnostic seeds)

| budget | training episodes | D BSS | F BSS | D − F, 95% CI | (1 − 2α) CI | reading |
|---|---|---|---|---|---|---|
| 1 n0 | 1,024 | +0.234 ± 0.033 | +0.224 ± 0.043 | +0.010 [-0.081, +0.101] | [-0.094, +0.115] | inconclusive |
| 2 n0 | 2,048 | +0.329 ± 0.035 | +0.301 ± 0.041 | +0.028 [-0.017, +0.073] | [-0.023, +0.080] | inconclusive |
| 4 n0 | 4,096 | +0.388 ± 0.037 | +0.379 ± 0.028 | +0.009 [-0.008, +0.025] | [-0.010, +0.028] | falsification (equivalent to zero within ±SESOI) |
| 8 n0 | 8,192 | +0.437 ± 0.032 | +0.441 ± 0.031 | -0.004 [-0.007, -0.001] | [-0.008, -0.001] | falsification (equivalent to zero within ±SESOI) |

## Inputs for the power analysis

With the observed SD of the paired difference (0.014 BSS), two-sided α = 0.05/3 and power 0.90, detecting an effect of size SESOI = 0.05 needs ((z₁₋α/₂ + z₀.₉₀) · SD / SESOI)² = **2 paired run(s)** (2 with the 80% upper confidence limit of SD). The manuscript's illustration assumed SD = 0.08 (≈ 35 runs). For H2, the per-seed λ slope of D has SD 0.015 BSS per unit λ; the D₁ − F₀ contrast has SD 0.017. For the equivalence test (two one-sided tests at α = 0.05/3 against ±0.05, power 0.90 when the true difference is 0) the requirement is ((z₁₋α + z₀.₉₅) · SD / SESOI)² = **2 paired run(s)** (2 with the 80% upper confidence limit of SD).

## Verdicts and automatically detected problems

- **H1 (pilot reading, not confirmatory):** falsification (equivalent to zero within ±SESOI); D − F = +0.002 BSS, (1 − 2α) CI [-0.009, +0.013].
- **H2 (pilot reading):** not informative: no D − F advantage at λ = 0.
- No automatic flags.

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

