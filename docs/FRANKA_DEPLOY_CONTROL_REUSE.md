# Read-only review of existing deployment controls

Source reviewed: `/data1/linsixu/Franka_Deploy`, 2026-09-09. No hardware
connection, deployment command or file modification was made in that workspace.

The user explicitly confirmed **FR3 + the original fully closed gripper** for
this pushing project. The deployment README's FR3 + Inspire RH56 configuration
belongs to other work. Its payload, end-effector frame and commissioning
profile must not be copied into this pushing task.

The current pushing simulation uses a Panda model. Existing Panda trials are
diagnostics, not final FR3 deployment acceptance. The next model alignment must
target FR3 and the stock gripper in both simulation and the OSC model; do not
continue treating the current Panda asset as the final physical target.

The maintained `robot_control/franka/backend.py` uses `JointPositions` with
`ControllerMode.JointImpedance`. Its persistent session, state validation,
payload/frame fields, watchdogs and command ownership are reusable boundaries.
It is not an existing native Push Anything torque/OSC backend.

`docs/controller_reference/franka_v258_libfranka_example.cpp` similarly sends
interpolated joint-position commands to the robot's internal joint impedance.
The repository describes these files as reference contracts, not the live
runtime owner.

`examples/run_eef_impedance_policy_env.py` contains an explicit Cartesian
impedance torque example: J-transpose times pose-error wrench, plus null-space
torque and Coriolis compensation. It does not implement Push Anything's
QP-based OSC. Presence of the example alone does not establish commissioning
or successful hardware validation of this pushing task.

The intended reuse is one control formulation and reference semantics, with
separate simulation/device adapters and explicit robot-model configuration.
Original Push Anything itself uses the same `franka_osc_controller` source
for simulation and hardware, with a hardware gravity-compensation removal
branch. Therefore raw simulation torques must not be forwarded unchanged to
the device without checking the driver's gravity convention.

Next work remains simulation-only: use the confirmed FR3/stock-gripper
contract, make dynamics/frame differences explicit, validate the original OSC
against that model, and preserve independent 20 Hz planner / fast servo
cadences. Real-time hardware execution cannot inherit the simulator's blocking
wait for a newly solved MPC trajectory.
