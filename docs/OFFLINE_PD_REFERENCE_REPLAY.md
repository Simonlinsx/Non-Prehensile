# Offline replay of captured PD references

Patch 0019 adds `--offline_pd_replay_input` to the native executable. It loads
the same scene model and contact pairs, reconstructs measured hand orientation
from the recorded seven Franka joint angles, evaluates a supplied full-precision
reference, and exits before constructing command publishers. This is a model
diagnostic, not another physical trial or an acceptance result. Normal execution
is unchanged when the flag is absent. The active physical evaluations continue
to use the separate archived `bde00e4...` executable.

The offline binary SHA256 is
`cc85afc2e93a77e24320d08e273491ffb79b6385eac8d8df4aef1c363791dda0`.
It is archived under `offline_pd_resolution_t8p3/native_binary`, together with
the patch/source snapshot. Build and patch reversibility checks passed.

`prepare_offline_pd_replay.py` extracts the first actually captured physical
reference, full-precision native state knots, published force knots and a
synchronized measured arm configuration. It requires exact equality between
the native actor state knots and executed actor position knots, matching knot
durations, C3 mode, force tracking and unchanged references through the replay.
Input and physical-source hashes are recorded. Six tests exercise the real
captured reference and reject mismatched positions, forces, timestamps and modes.

The native audit calls the same relinearized 1 kg PD model used by the calibrated
cost. `analyze_offline_pd_resolution.py` checks the existing physical-replay
eligibility report, positive legal contact, zero shielding/C1 violations, exact
physical endpoint time and original-resolution equivalence before comparing
alternate integration resolutions. Both original-resolution predictions match
all **11 × 19 state values exactly** against the online diagnostic, including
the second window's independently verified same-plan baseline.

| Fine time step | XY error, 8.300–8.600 s | SO(3) error | XY error, 8.350–8.650 s | SO(3) error |
| --- | ---: | ---: | ---: | ---: |
| 18.750 ms (original) | 6.708 mm | 0.045254 rad | 9.531 mm | 0.003536 rad |
| 9.375 ms | 8.468 mm | 0.049285 rad | 8.889 mm | 0.010475 rad |
| 5.000 ms | 9.763 mm | 0.056374 rad | 8.396 mm | 0.012904 rad |
| 2.500 ms | 10.422 mm | 0.060876 rad | 9.427 mm | 0.018727 rad |

Finer integration does not consistently improve the physical prediction. It
worsens both errors in the first window and all rotation errors in the second.
The two windows overlap, so they are not independent scene validation. The
active controller retains its original resolution of four fine steps per
75 ms knot. Each offline comparison takes approximately 0.2–1.2 s on CPU.

Reports and exact inputs are under
`outputs/contact_planner_m3/workspace_height_20260908/offline_pd_resolution_t8p3`
and `offline_pd_resolution_t8p35`; each contains `comparison.json` and
`input_provenance.json`. Reproduce extraction and analysis with:

```bash
python scripts/prepare_offline_pd_replay.py \
  --scene-directory <completed-reference-replay/scene007> \
  --output-directory <new-offline-output>
# Run the recorded native command in each isolated runtime copy.
python scripts/analyze_offline_pd_resolution.py \
  --directory <offline-output> --baseline-native-log <online-calibration.log>
```

# Sampled reference timing follow-up

The revised offline-only patch 0019 adds `--offline_pd_sampled_references=true`.
It models 20 Hz packets, 100 Hz position extrapolation and reference holding,
with PD feedback at each integration step. It does not change active execution
or the calibrated cost. The new binary and patch are archived in
`offline_pd_sampled_t8p3` and `offline_pd_sampled_t8p35`.

All available historical raw commands match: 14 samples in the first window,
15 in the second, with maximum force difference below 8e-15 N. Missing trace
times remain explicitly unverified. Applied positions differ from the raw
model by up to 0.734 mm / 0.307 mm due to the execution governor, which is not
included in this model. Both original-resolution baselines still reproduce
all online predicted states exactly.

| Sampled model fine step | First window XY / SO(3) error | Second window XY / SO(3) error |
| --- | ---: | ---: |
| 5 ms | 7.842 mm / 0.052624 rad | 8.197 mm / 0.017116 rad |
| 2.5 ms | 8.998 mm / 0.061486 rad | 8.488 mm / 0.017393 rad |
| 1.25 ms | 9.561 mm / 0.062480 rad | 8.318 mm / 0.017286 rad |

At matching 5 / 2.5 ms integration steps, sampled timing improves XY predictions
in both windows, but rotation does not consistently improve. It does not beat
the original 18.75 ms baseline across these pose metrics. This is insufficient
evidence to promote it into active cost. Eleven extraction/reference-clock
tests, the native build and all patch reverse checks pass. Detailed reports:
`docs/evidence/contact_planner_m3_offline_sampled_pd_20260908.json`.

The subsequent measured-tensor and LCP consistency study is recorded in
[OFFLINE_PD_INERTIA_AND_LCP.md](OFFLINE_PD_INERTIA_AND_LCP.md).

# Completed debug failure

The frozen cost-only controller's front-direction debug `scene000` ended at
74.31 s with watchdog timeout after the native workspace-radius assertion:
measured actor radius **0.750472 m**, upper limit **0.75 m**. Final XY error was
**288.930 mm**, SO(3) **0.354046 rad**, zero dwell and zero C1 violations. This
is a failed task and incomplete 180 s execution, not a successful safety-only
trial. It remains in the three-scene debug batch; subsequent scenes continue.

An auxiliary check found three current-cost-infinite records still coinciding
with C3 mode and finite alternative costs. No unshielded nonzero force was
observed in the retained samples of those intervals. This does not establish
the cause of the drift; no mode-switching change has been made on that basis.

# Lossless storage of older debugging logs

Four completed older 180 s trials now store `effect_audit.jsonl.gz` plus
`effect_audit_archive.json`: `nonnegative_quat10_video180`,
`semantic_quat_fixed_q10_video180`, `native_osc_guard_q100_video180`, and
`fixed_goal_rot100_q100_video180`. Original and compressed SHA256 values are
recorded and decompression was verified before replacing each plain log. One
real restoration was also verified. Use
`python scripts/archive_completed_effect_audits.py restore --metadata <scene/effect_audit_archive.json>`
before rerunning an older analysis that expects the uncompressed pathname.
The original hashes in historical reports still identify the restored bytes.
