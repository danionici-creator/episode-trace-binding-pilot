# Simulation pilot — Does Explicit Episode–Trace Binding Help?

Code, frozen configurations, per-run results, and reports of the five exploratory pilot iterations
described in the preprint

> Ioan Daniel Ionici (2026). *Does Explicit Episode–Trace Binding Help? An Exploratory Study with
> Information-Matched, Competence-Audited Controls. Pilot Evidence, Open Problems, and a Protocol for Testing
> Structural Priors.* Preprint, version 1.3, 1 November 2026. https://doi.org/10.5281/zenodo.23666335. ORCID 0009-0009-8183-3656.

Code: MIT License (`LICENSE`). Configurations, results, reports, and documentation: CC BY 4.0
(`LICENSE-DATA.md`). How to cite: `CITATION.cff`. All results are exploratory; no confirmatory data
have been collected.

## Status and scope of verification

This project is the independent contribution of one author, who used generative-AI systems extensively to turn
an intuition into a testable question, an exploratory pilot, a manuscript, and this archive. The results are
exploratory and have not been independently validated: the reported numbers are recomputed from the saved
per-run outputs, and selected properties of the pipeline (pairing, permutations, data splits, seed isolation,
determinism) are checked by automated tests and a local re-run; no independent human code review or retraining
has been carried out, and these lie outside the defined scope of this project. Nothing here asks to be accepted on
trust. The archive is released as it stands so that anyone interested can examine, rerun, correct, or extend it;
readers who regard independent human verification as a prerequisite should treat the empirical findings as
provisional and may assess the formulation, the method, and the open problems on their own merits.

> **Working titles in the artifacts.** The design notes and reports were written while the study was
> drafted under earlier working titles ("Phase I simulation pilot", "Study 1", *Binding Episodes to Their
> Internal Effects*) and call the manuscript simply "the manuscript". They refer to the preprint above;
> they are left unchanged for the reasons given in the next note.
>
> **Terminology note (added 25 October 2026, after the pilots were completed).** Throughout these
> artifacts — configuration files, design notes, reports, and code comments — the word
> "preregistered" means *fixed before the main seeds were simulated* (an internal pre-specification).
> No part of the pilot was publicly preregistered; the manuscript reports all five iterations as
> exploratory and lists every design decision in its Section 4.6 and Table 4. The original files are
> left unchanged so that their recorded SHA-256 hashes and the byte-identical re-rendering of the
> reports remain valid.
>
> **Reproducing the manuscript tables.** `python pilot/reproduce_manuscript_tables.py` (Python and
> NumPy only) recomputes every contrast reported in Tables 5, 7, 8 and 9 and Figure 2 of the preprint (versions 1.2 and 1.3)
> from the per-run scores in the result files, and checks the SHA-256 hashes of the frozen iteration-4 and
> iteration-5 configurations.
> This establishes the internal consistency of the reported numbers; it is not a substitute for
> independent retraining, which has not yet been carried out.
>
> **Statistical terminology.** All intervals use the fixed Bonferroni-adjusted level α = 0.05/3 for the
> family of three hypotheses (`alpha_two_sided` in the configuration files). The preprint calls the
> "co-timed" objective of the code and reports "join-supervised": it is label-free with respect to provenance
> but supplies the correct record-signature pairing as a training signal.
>
> **Iteration 5 code change.** `train_sm.py` gained one option, `training.aux_prebind` (absent, i.e. false,
> in every earlier configuration, so iterations 1-4 are unchanged): the co-timed target is then predicted from a
> second encoder pass in which D's bound copy is zeroed.
>
> **One command.** `python pilot/reproduce_lite.py` runs three levels of verification in turn: the tables from the
> saved results, the audit checks, and a smoke training of D and F on two seeds (about 7 minutes in all on six
> workers); `--tables` runs the first only, with NumPy alone. **Environment:** `pip install --require-hashes -r
> pilot/requirements.lock` installs the exact versions used, with hashes (version 1.3 of the archive).
>
> **Levels of reproduction.** Python and NumPy only: `reproduce_manuscript_tables.py`, `supplementary_analyses.py`,
> `make_run_inventory.py` (internal consistency of the reported numbers). With the recorded PyTorch environment:
> `audit_checks.py` and `iter5_run.py --report-only`. With hours of CPU time: every training run. All runs used one
> Windows 11 machine with six worker processes of two threads each; with the same thread settings re-runs reproduce the
> recorded scores exactly, with other thread counts decoder results differ by up to 0.002 BSS. No container image is
> provided.
>
> **Run inventory and environment.** `python pilot/make_run_inventory.py` writes `runs_inventory.csv`, one row
> per trained run (all runs of the five iterations: iteration, condition, architecture, lambda, budget,
> seed, test BSS). `requirements.txt` lists the exact package versions used (Python 3.14.6, NumPy 2.5.1,
> SciPy 1.18.0, PyTorch 2.13.0 CPU, PyYAML 6.0.3).


Minimal instantiation of the study: a persistent synthetic
environment with three provenance routes through one plastic pathway, a path-dependent slow
state, the preregistered measurement M and signature Phi, the binding architecture D and the
information control F, the misalignment manipulation lambda, and the manipulation checks M1, M2,
M3 and the leakage audit.

```
python pilot/run_pilot.py               # iteration 1, ~35 min on CPU -> results.json, REPORT.md
python pilot/run_pilot.py --quick       # smoke test, about 5 minutes on six workers -> quick/
python pilot/run_pilot.py --report-only # re-render REPORT.md from results.json
python pilot/calibrate_env.py measurement.sigma_z=0.25   # M2/M1 on the tuning seed only

python pilot/iter2_candidates.py        # iteration 2: M4 candidate search, tuning seed only -> candidates_iter2.json
python pilot/iter2_candidates.py --repro-only --screen-rest   # iteration-1 reproduction check; post-selection screen
python pilot/run_pilot.py --config config_iter2.yaml          # iteration 2 main run -> results_iter2.json, REPORT_iter2.md

python pilot/iter3_candidates.py        # iteration 3: strongest admissible F, tuning seed only -> candidates_iter3.json
python pilot/run_pilot.py --config config_iter3.yaml          # iteration 3 main run -> results_iter3.json, REPORT_iter3.md

python pilot/iter4_tune.py              # iteration 4: tune G8, T32, G32 on seed 1000 -> tuning_iter4.json
python pilot/iter4_extend.py            # iteration 4: plateau-rule amendment (learning onset), tuning seed only
python pilot/iter4_run.py               # iteration 4 main run on fresh seeds 11-20 -> results_iter4.json, REPORT_iter4.md

python pilot/iter5_run.py               # iteration 5 (no tuning; resumable) -> results_iter5*.json(l), REPORT_iter5.md
python pilot/iter5_diagnostics.py       # construct-validity diagnostics -> results_iter5_diagnostics.json
python pilot/supplementary_analyses.py  # analyses of existing results -> results_supplementary.json
python pilot/audit_checks.py            # automated audit of the pipeline (exit code = number of failures)
```

Iteration 2 adds precondition M4 (control competence) and the iteration-2 options (curriculum,
co-timed auxiliary loss), all behind config flags; with `config.yaml` the code reproduces iteration 1
exactly.

Requirements: Python 3, numpy, scipy, PyYAML, PyTorch (CPU); exact versions in `requirements.txt`
(`pip install -r pilot/requirements.txt`). Commands are run from the folder that contains `pilot/`.

| File | Content |
|---|---|
| `config.yaml` | every preregistered setting: environment, M, Phi, architectures, tuning grid, conditions, checks |
| `sim_env.py` | environment, routes, plastic agent h = (W, U), M and Phi, M1 check, life simulation |
| `checks.py` | M2 decoding-ceiling decoders (called "information ceiling" in the code), control task, misaligned control, leakage decoder |
| `self_models.py` | shared transformer backbone; D (binding) and F (control); width compensation |
| `train_sm.py` | one training/evaluation run; D_lambda / F_lambda misbinding; F_noz |
| `scoring.py` | multiclass Brier skill score, log loss, accuracy, ECE |
| `run_pilot.py` | orchestration, paired analysis, power inputs, report |
| `DESIGN_NOTES.md` | design problems found while instantiating the protocol (included in the report) |
| `results.json` | all numbers, per seed and per condition, plus the frozen config |
| `cache/per_episode_brier.npz` | per-episode Brier scores of every run (for a mixed model / simulation-based power) |
| `config_iter2.yaml` | iteration 2: frozen configuration (selected candidate a), M4, candidate definitions |
| `iter2_candidates.py` | iteration 2: M4 candidate search on the tuning seed, in the preregistered order |
| `candidates_iter2.json` | iteration 2: M4 for every candidate tried, plateau ladder, reproduction check |
| `DESIGN_NOTES_iter2.md` | iteration 2: procedure and what the candidate screen shows (included in REPORT_iter2.md) |
| `results_iter2.json`, `REPORT_iter2.md`, `cache/per_episode_brier_iter2.npz` | iteration 2 outputs |
| `config_iter3.yaml` | iteration 3: frozen configuration (a+c, 4800 steps), strengthened M4, shared objective |
| `iter3_candidates.py`, `candidates_iter3.json` | iteration 3: candidate screen with per-candidate plateau ladders, M4, selection |
| `DESIGN_NOTES_iter3.md` | iteration 3: procedure and screening findings (included in REPORT_iter3.md) |
| `results_iter3.json`, `REPORT_iter3.md`, `cache/per_episode_brier_iter3.npz` | iteration 3 outputs |
| `config_iter4.yaml`, `config_iter4.freeze.json` | iteration 4: frozen configuration (conditions R, G8, T32, G32) and its SHA-256 freeze record |
| `iter4_common.py`, `iter4_tune.py`, `iter4_extend.py`, `tuning_iter4.json` | iteration 4: condition configs, tuning on seed 1000, plateau amendment |
| `iter4_run.py`, `results_iter4.json`, `REPORT_iter4.md`, `cache/per_episode_brier_iter4.npz` | iteration 4: main run on fresh seeds and outputs |
| `DESIGN_NOTES_iter4.md` | iteration 4: procedure, tuning findings, deviations (included in REPORT_iter4.md) |

| `reproduce_manuscript_tables.py` | recomputes Tables 5, 7, 8 and 9 and Figure 2 of the preprint (v1.2 and v1.3) from the result files; checks the iteration-4 and iteration-5 hashes |
| `reproduce_lite.py`, `requirements.lock` | one command for three levels of verification (tables, audit checks, smoke training); the exact environment with hashes |
| `supplementary_analyses.py`, `results_supplementary.json` | onset times, margin and competence-threshold sensitivity, leakage calibration (no training) |
| `audit_checks.py` | 20 automated checks of pairing, permutations, splits, alignment, isolation, determinism, and of where the co-timed objective reads (about a minute; requires PyTorch) |
| `config_iter5.yaml`, `config_iter5.freeze.json`, `DESIGN_NOTES_iter5.md` | iteration 5: frozen configuration, freeze record, a priori readings |
| `iter5_run.py`, `results_iter5_runs.jsonl`, `results_iter5_lives.json`, `results_iter5.json`, `REPORT_iter5.md` | iteration 5: diagnostic of the K = 32 crossover and re-run audit (resumable) |
| `iter5_diagnostics.py`, `results_iter5_diagnostics.json` | construct-validity diagnostics: decoders on reduced inputs and on lives with the slow state reset |
| `iter5_rerun_episodes.py`, `results_iter5_rerun_check.json`, `cache/per_episode_brier_iter5.npz` | iteration 5 re-run in full: all 100 test scores reproduced exactly (max |diff| 0); per-episode Brier scores, as for iterations 1-4 |
| `environment.txt` | `pip freeze` of the environment in which every iteration ran |
| `make_run_inventory.py`, `runs_inventory.csv` | one row per trained run (all iterations) |
| `requirements.txt` | exact package versions |

Seeds: 1000 is used only for calibration and tuning; seeds 1-10 are the main seeds of iterations 1-3
(iteration 3 reused the agent lives of iteration 2) and seeds 11-20 those of iterations 4 and 5. Each seed is a
new environment instance and a new agent life; D and F are paired within seed (same data, same
batch order).
