"""Versioned binary contract for an online C3+ simulator/hardware bridge.

The transport is deliberately simulator independent.  Isaac Lab and the real
robot publish the same measured state packet; a small native relay translates
it to the LCM messages consumed by Push Anything and returns the latest joint
effort command.  All values use SI units and the fixed Franka joint order
``panda_joint1`` through ``panda_joint7``.

Object poses are expressed in the C3 planner world frame.  For the supported
DOMINO mesh this means removing the baked support rotation from the Isaac or
camera pose and subtracting the table-frame Z offset before packing a packet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import math
import struct
from typing import Sequence


PROTOCOL_VERSION = 2
SCENE_PROTOCOL_VERSION = 3
MAX_CLUTTER_OBJECTS = 3
STATE_MAGIC = b"C3ST"
SCENE_STATE_MAGIC = b"C3SC"
COMMAND_MAGIC = b"C3CM"
TASK_COMMAND_MAGIC = b"C3TK"
_STATE_STRUCT = struct.Struct("!4sHHQq34d")
# Robot q/dq/tau (21), target rigid-body state (13), then three fixed clutter
# slots (39).  The active count is explicit and inactive slots have canonical
# identity/zero padding, keeping framing deterministic on a raw TCP stream.
_SCENE_STATE_STRUCT = struct.Struct("!4sHHQqH73d")
_COMMAND_STRUCT = struct.Struct("!4sHHQq7d")
_TASK_COMMAND_STRUCT = struct.Struct("!4sHHQq9d")
STATE_PACKET_SIZE = _STATE_STRUCT.size
SCENE_STATE_PACKET_SIZE = _SCENE_STATE_STRUCT.size
COMMAND_PACKET_SIZE = _COMMAND_STRUCT.size
TASK_COMMAND_PACKET_SIZE = _TASK_COMMAND_STRUCT.size


class C3CommandFlags(IntFlag):
    """Controller state returned with every effort command."""

    READY = 1 << 0
    STALE_STATE = 1 << 1
    SEMANTIC_HOLD = 1 << 2
    PLANNER_FAILURE = 1 << 3
    C3_MODE = 1 << 4
    # The active C3 contact action is explicitly intended to reduce target
    # yaw error.  This bit is meaningful only together with ``C3_MODE`` and
    # lets a robot-specific executor preserve the planner's action semantics
    # across contact acquisition, bounded push pulses, and physical retreat.
    YAW_RECOVERY = 1 << 5


class C3StateFlags(IntFlag):
    """Measured facts and explicit mode requests sent to the C3 relay."""

    # A physical contact sensor fired and semantic geometry classified the
    # contact as target-safe.  This remains an observation when force-C3 is
    # disabled, so contact-effect audits do not depend on controller policy.
    LEGAL_SAFE_CONTACT = 1 << 0
    # Explicitly request Push Anything's native force-C3 radio mode.  Keeping
    # this separate from LEGAL_SAFE_CONTACT prevents a diagnostic observation
    # from silently changing planner behavior.
    FORCE_C3_MODE = 1 << 1


def _validate_state_flags(flags: C3StateFlags) -> int:
    raw_flags = int(flags)
    known_flags = int(
        C3StateFlags.LEGAL_SAFE_CONTACT | C3StateFlags.FORCE_C3_MODE
    )
    if raw_flags & ~known_flags:
        raise ValueError("state contains unknown status flags")
    return raw_flags


def _finite_tuple(
    values: Sequence[float], length: int, field_name: str
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{field_name} must contain {length} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field_name} must contain only finite values")
    return result


def _validate_header(sequence: int, utime_us: int) -> tuple[int, int]:
    sequence = int(sequence)
    utime_us = int(utime_us)
    if not 0 <= sequence < 2**64:
        raise ValueError("sequence must fit an unsigned 64-bit integer")
    if not 0 <= utime_us < 2**63:
        raise ValueError("utime_us must be non-negative and fit int64")
    return sequence, utime_us


@dataclass(frozen=True)
class C3RigidBodyState:
    """Pose and spatial velocity for one object in the C3 world frame."""

    quaternion_wxyz: tuple[float, ...]
    position_m: tuple[float, ...]
    angular_velocity_rad_s: tuple[float, ...]
    linear_velocity_m_s: tuple[float, ...]

    def values(self, field_prefix: str = "object") -> tuple[float, ...]:
        quaternion = _finite_tuple(
            self.quaternion_wxyz, 4, f"{field_prefix}_quaternion_wxyz"
        )
        quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
        if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-3):
            raise ValueError(f"{field_prefix}_quaternion_wxyz must be normalized")
        return (
            *quaternion,
            *_finite_tuple(self.position_m, 3, f"{field_prefix}_position_m"),
            *_finite_tuple(
                self.angular_velocity_rad_s,
                3,
                f"{field_prefix}_angular_velocity_rad_s",
            ),
            *_finite_tuple(
                self.linear_velocity_m_s,
                3,
                f"{field_prefix}_linear_velocity_m_s",
            ),
        )

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "C3RigidBodyState":
        values = _finite_tuple(values, 13, "rigid_body_state")
        return cls(
            quaternion_wxyz=tuple(values[0:4]),
            position_m=tuple(values[4:7]),
            angular_velocity_rad_s=tuple(values[7:10]),
            linear_velocity_m_s=tuple(values[10:13]),
        )


_INACTIVE_RIGID_BODY_VALUES = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


@dataclass(frozen=True)
class C3MeasuredSceneState:
    """Measured Franka, target, and active clutter state for S3 planning.

    The target is always slot zero.  ``clutter`` follows the staged C3
    ``base_names`` and LCM channel order.  A fixed maximum keeps packet
    framing simple enough for both Isaac and a future hardware bridge.
    """

    sequence: int
    utime_us: int
    joint_position_rad: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]
    joint_effort_nm: tuple[float, ...]
    target: C3RigidBodyState
    clutter: tuple[C3RigidBodyState, ...] = ()
    flags: C3StateFlags = C3StateFlags(0)

    def pack(self) -> bytes:
        sequence, utime_us = _validate_header(self.sequence, self.utime_us)
        position = _finite_tuple(self.joint_position_rad, 7, "joint_position_rad")
        velocity = _finite_tuple(
            self.joint_velocity_rad_s, 7, "joint_velocity_rad_s"
        )
        effort = _finite_tuple(self.joint_effort_nm, 7, "joint_effort_nm")
        clutter = tuple(self.clutter)
        if len(clutter) > MAX_CLUTTER_OBJECTS:
            raise ValueError(
                f"clutter supports at most {MAX_CLUTTER_OBJECTS} objects"
            )
        body_values = list(self.target.values("target"))
        for index, state in enumerate(clutter):
            if not isinstance(state, C3RigidBodyState):
                raise TypeError("clutter entries must be C3RigidBodyState")
            body_values.extend(state.values(f"clutter_{index}"))
        for _ in range(MAX_CLUTTER_OBJECTS - len(clutter)):
            body_values.extend(_INACTIVE_RIGID_BODY_VALUES)
        return _SCENE_STATE_STRUCT.pack(
            SCENE_STATE_MAGIC,
            SCENE_PROTOCOL_VERSION,
            _validate_state_flags(self.flags),
            sequence,
            utime_us,
            len(clutter),
            *position,
            *velocity,
            *effort,
            *body_values,
        )

    @classmethod
    def unpack(cls, packet: bytes) -> "C3MeasuredSceneState":
        if len(packet) != SCENE_STATE_PACKET_SIZE:
            raise ValueError(
                f"scene state packet must contain {SCENE_STATE_PACKET_SIZE} bytes"
            )
        (
            magic,
            version,
            flags,
            sequence,
            utime_us,
            clutter_count,
            *payload,
        ) = _SCENE_STATE_STRUCT.unpack(packet)
        if (
            magic != SCENE_STATE_MAGIC
            or version != SCENE_PROTOCOL_VERSION
            or flags & ~int(
                C3StateFlags.LEGAL_SAFE_CONTACT | C3StateFlags.FORCE_C3_MODE
            )
        ):
            raise ValueError("unsupported or malformed scene state packet header")
        if clutter_count > MAX_CLUTTER_OBJECTS:
            raise ValueError("scene state packet has an invalid clutter count")
        body_payload = payload[21:]
        bodies = [
            C3RigidBodyState.from_values(body_payload[start : start + 13])
            for start in range(0, len(body_payload), 13)
        ]
        for index in range(1 + clutter_count, 1 + MAX_CLUTTER_OBJECTS):
            if tuple(body_payload[index * 13 : (index + 1) * 13]) != (
                _INACTIVE_RIGID_BODY_VALUES
            ):
                raise ValueError("inactive clutter slots must use canonical padding")
        state = cls(
            sequence=sequence,
            utime_us=utime_us,
            joint_position_rad=tuple(payload[0:7]),
            joint_velocity_rad_s=tuple(payload[7:14]),
            joint_effort_nm=tuple(payload[14:21]),
            target=bodies[0],
            clutter=tuple(bodies[1 : 1 + clutter_count]),
            flags=C3StateFlags(flags),
        )
        state.pack()
        return state


@dataclass(frozen=True)
class C3MeasuredState:
    """One measured robot/object state sample sent to the C3+ controller."""

    sequence: int
    utime_us: int
    joint_position_rad: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]
    joint_effort_nm: tuple[float, ...]
    object_quaternion_wxyz: tuple[float, ...]
    object_position_m: tuple[float, ...]
    object_angular_velocity_rad_s: tuple[float, ...]
    object_linear_velocity_m_s: tuple[float, ...]
    flags: C3StateFlags = C3StateFlags(0)

    def pack(self) -> bytes:
        sequence, utime_us = _validate_header(self.sequence, self.utime_us)
        position = _finite_tuple(self.joint_position_rad, 7, "joint_position_rad")
        velocity = _finite_tuple(
            self.joint_velocity_rad_s, 7, "joint_velocity_rad_s"
        )
        effort = _finite_tuple(self.joint_effort_nm, 7, "joint_effort_nm")
        quaternion = _finite_tuple(
            self.object_quaternion_wxyz, 4, "object_quaternion_wxyz"
        )
        quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
        if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-3):
            raise ValueError("object_quaternion_wxyz must be normalized")
        object_position = _finite_tuple(
            self.object_position_m, 3, "object_position_m"
        )
        angular_velocity = _finite_tuple(
            self.object_angular_velocity_rad_s,
            3,
            "object_angular_velocity_rad_s",
        )
        linear_velocity = _finite_tuple(
            self.object_linear_velocity_m_s, 3, "object_linear_velocity_m_s"
        )
        payload = (
            *position,
            *velocity,
            *effort,
            *quaternion,
            *object_position,
            *angular_velocity,
            *linear_velocity,
        )
        return _STATE_STRUCT.pack(
            STATE_MAGIC,
            PROTOCOL_VERSION,
            _validate_state_flags(self.flags),
            sequence,
            utime_us,
            *payload,
        )

    @classmethod
    def unpack(cls, packet: bytes) -> "C3MeasuredState":
        if len(packet) != STATE_PACKET_SIZE:
            raise ValueError(
                f"state packet must contain {STATE_PACKET_SIZE} bytes"
            )
        magic, version, flags, sequence, utime_us, *payload = _STATE_STRUCT.unpack(
            packet
        )
        if (
            magic != STATE_MAGIC
            or version != PROTOCOL_VERSION
            or flags & ~int(
                C3StateFlags.LEGAL_SAFE_CONTACT | C3StateFlags.FORCE_C3_MODE
            )
        ):
            raise ValueError("unsupported or malformed state packet header")
        state = cls(
            sequence=sequence,
            utime_us=utime_us,
            joint_position_rad=tuple(payload[0:7]),
            joint_velocity_rad_s=tuple(payload[7:14]),
            joint_effort_nm=tuple(payload[14:21]),
            object_quaternion_wxyz=tuple(payload[21:25]),
            object_position_m=tuple(payload[25:28]),
            object_angular_velocity_rad_s=tuple(payload[28:31]),
            object_linear_velocity_m_s=tuple(payload[31:34]),
            flags=C3StateFlags(flags),
        )
        # Reuse pack-time validation for values received from the network.
        state.pack()
        return state


@dataclass(frozen=True)
class C3EffortCommand:
    """One C3+/OSC joint-effort command returned to the executor."""

    sequence: int
    utime_us: int
    flags: C3CommandFlags
    joint_effort_nm: tuple[float, ...]

    def pack(self) -> bytes:
        sequence, utime_us = _validate_header(self.sequence, self.utime_us)
        effort = _finite_tuple(self.joint_effort_nm, 7, "joint_effort_nm")
        raw_flags = int(self.flags)
        known_flags = int(
            C3CommandFlags.READY
            | C3CommandFlags.STALE_STATE
            | C3CommandFlags.SEMANTIC_HOLD
            | C3CommandFlags.PLANNER_FAILURE
            | C3CommandFlags.C3_MODE
            | C3CommandFlags.YAW_RECOVERY
        )
        if raw_flags & ~known_flags:
            raise ValueError("command contains unknown status flags")
        return _COMMAND_STRUCT.pack(
            COMMAND_MAGIC,
            PROTOCOL_VERSION,
            raw_flags,
            sequence,
            utime_us,
            *effort,
        )

    @classmethod
    def unpack(cls, packet: bytes) -> "C3EffortCommand":
        if len(packet) != COMMAND_PACKET_SIZE:
            raise ValueError(
                f"command packet must contain {COMMAND_PACKET_SIZE} bytes"
            )
        magic, version, raw_flags, sequence, utime_us, *effort = (
            _COMMAND_STRUCT.unpack(packet)
        )
        if magic != COMMAND_MAGIC or version != PROTOCOL_VERSION:
            raise ValueError("unsupported or malformed command packet header")
        command = cls(
            sequence=sequence,
            utime_us=utime_us,
            flags=C3CommandFlags(raw_flags),
            joint_effort_nm=tuple(effort),
        )
        # Reuse pack-time validation, including rejection of unknown bits.
        command.pack()
        return command


@dataclass(frozen=True)
class C3TaskCommand:
    """Current task-space sample from C3's online execution trajectory.

    The simulator or robot-specific low-level controller tracks this command,
    avoiding direct reuse of model-dependent Drake joint torques.
    """

    sequence: int
    utime_us: int
    flags: C3CommandFlags
    position_m: tuple[float, ...]
    velocity_m_s: tuple[float, ...]
    feedforward_force_n: tuple[float, ...]

    def pack(self) -> bytes:
        sequence, utime_us = _validate_header(self.sequence, self.utime_us)
        position = _finite_tuple(self.position_m, 3, "position_m")
        velocity = _finite_tuple(self.velocity_m_s, 3, "velocity_m_s")
        force = _finite_tuple(
            self.feedforward_force_n, 3, "feedforward_force_n"
        )
        raw_flags = int(self.flags)
        known_flags = int(
            C3CommandFlags.READY
            | C3CommandFlags.STALE_STATE
            | C3CommandFlags.SEMANTIC_HOLD
            | C3CommandFlags.PLANNER_FAILURE
            | C3CommandFlags.C3_MODE
            | C3CommandFlags.YAW_RECOVERY
        )
        if raw_flags & ~known_flags:
            raise ValueError("task command contains unknown status flags")
        return _TASK_COMMAND_STRUCT.pack(
            TASK_COMMAND_MAGIC,
            PROTOCOL_VERSION,
            raw_flags,
            sequence,
            utime_us,
            *position,
            *velocity,
            *force,
        )

    @classmethod
    def unpack(cls, packet: bytes) -> "C3TaskCommand":
        if len(packet) != TASK_COMMAND_PACKET_SIZE:
            raise ValueError(
                f"task command packet must contain {TASK_COMMAND_PACKET_SIZE} bytes"
            )
        magic, version, raw_flags, sequence, utime_us, *payload = (
            _TASK_COMMAND_STRUCT.unpack(packet)
        )
        if magic != TASK_COMMAND_MAGIC or version != PROTOCOL_VERSION:
            raise ValueError("unsupported or malformed task command header")
        command = cls(
            sequence=sequence,
            utime_us=utime_us,
            flags=C3CommandFlags(raw_flags),
            position_m=tuple(payload[0:3]),
            velocity_m_s=tuple(payload[3:6]),
            feedforward_force_n=tuple(payload[6:9]),
        )
        command.pack()
        return command
