# Native OSC torque baseline

User clarification after these tests: the intended hardware is **FR3 with its
original gripper fully closed**. The Panda holds below remain diagnostic
evidence. Subsequent model alignment must use FR3 assets and its stock gripper,
rather than freezing the measured Panda dynamics as the final deployment
model. See `FRANKA_DEPLOY_CONTROL_REUSE.md`.

The user's requested next direction is to reuse the original Push Anything
trajectory generators and OSC, with IsaacLab receiving the resulting joint
torques. The earlier scalar-PD and Isaac OSC parameter comparisons are no
longer the main next action. Existing trials retain their results.

The existing effort executor used an attached sphere and forced C3 mode after
safe contact. The new optional stock-gripper path uses the unchanged stock
asset, commands both fingers closed, and no longer forces C3 on contact.
Robot gravity is enabled and arm position drives are disabled. Native
simulation OSC supplies gravity compensation; Isaac adds none. The relay
waits for the torque at the current measured timestamp, rejecting old or
future torques beyond 100 us. Task-mode launch defaults are unchanged.

## Completed stationary integration test

`outputs/contact_planner_m3/workspace_height_20260908/native_osc_closed_hold2`
uses the upstream stationary teleop mode, 1 ms physical/control steps, and
2 seconds of simulation. This is not a pushing trial or acceptance result.

- 2,000/2,000 fresh torque commands, zero stale commands or watchdogs.
- Maximum timestamp discrepancy: 1 us.
- Maximum commanded/applied torque discrepancy: 9.532e-7 Nm (float32 rounding).
- Terminal task-point displacement: 9.006 mm; maximum displacement: 10.041 mm.
- Maximum hand orientation change from first post-step sample: 0.023765 rad.
- Maximum absolute finger joint coordinate: 0.142 mm; commanded closed.
- No recorded C1 violation. No legal pushing contact or task success claimed.

`native_osc_closed_hold2_dynamics` repeats the same test with a read-only initial
PhysX mass/COM/inertia/gravity/mass-matrix export. **All 2,000 trace rows,
including every commanded and applied torque, are exactly equal.**

## Direct robot-model mismatch

Patch 0023 adds a default-disabled offline input to the native OSC executable.
It constructs the same `AddFrankaToPlant` model and exports gravity, mass matrix,
tip pose and body masses before creating LCM or any control diagram. The
initial joint vector is exactly the measured Isaac vector used by both tests.

Native and Isaac gravity-compensation torques differ by **3.450501 Nm** at
joint 3 and **1.276330 Nm** at joint 5. The mismatch is present before object
contact, independently of friction or contact sampling.

The robot assets differ throughout the arm, not just at the end effector:

| Body | Native model mass (kg) | Measured Isaac mass (kg) |
| --- | ---: | ---: |
| panda_link1 | 2.740000 | 2.360000 |
| panda_link3 | 2.380000 | 2.649882 |
| panda_link6 | 1.550000 | 1.128581 |
| panda_link8 | 0 | 1.000000 |

The native attached tool has total mass 0.269 kg. Isaac's hand and fingers
total approximately 0.586441 kg, in addition to its separate link8 body.
Isaac's measured arm masses agree with its packaged `panda_arm_hand.urdf`.
That source omits link8's inertial element; its measured 1 kg therefore needs
explicit treatment during model alignment. This evidence does not establish
that gravity mismatch alone causes all of the holding offset.

Full records and binary/source hashes are in each trial directory,
`integration_audit.json`, `robot_model_comparison.json`, and the second trial's
`offline_native_binary/` archive. The online tests use OSC SHA256
`91361234eae14156833d6161b8c51eac1b3d0c1f73718c6ed31dced6aed79f90`.

## Next action before pushing

Create an isolated native full-robot dynamics model matching the measured
stock-closed-gripper geometry and inertias. Compare FK, gravity and mass matrix
at identical joint states, then repeat the stationary test with unchanged OSC
gains. Avoid compensating this discrepancy with another gain sweep.

Also separate the original OSC's 1 kHz measured-state channel from the MPC's
20 Hz state channel before a task trial. The current hold diagnostic shares
the state channel and therefore is not an audited 20 Hz MPC execution.
The torque executor remains explicitly ineligible for formal acceptance until
the fixed-goal, continuous-pose, source and C1 audits are brought to parity.
