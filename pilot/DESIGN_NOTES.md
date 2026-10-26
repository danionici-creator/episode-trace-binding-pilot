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
