from __future__ import annotations

import math
import struct
import unittest

from dapl.contact_planner import (
    C3CommandFlags,
    C3EffortCommand,
    C3MeasuredSceneState,
    C3MeasuredState,
    C3RigidBodyState,
    C3StateFlags,
    C3TaskCommand,
    COMMAND_PACKET_SIZE,
    MAX_CLUTTER_OBJECTS,
    SCENE_STATE_PACKET_SIZE,
    STATE_PACKET_SIZE,
    TASK_COMMAND_PACKET_SIZE,
)


class C3OnlineProtocolTest(unittest.TestCase):
    @staticmethod
    def rigid_body(x: float) -> C3RigidBodyState:
        return C3RigidBodyState(
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            position_m=(x, 0.2, 0.01),
            angular_velocity_rad_s=(0.0, 0.0, 0.1),
            linear_velocity_m_s=(0.01, 0.0, 0.0),
        )

    def test_state_round_trip_has_fixed_size(self) -> None:
        state = C3MeasuredState(
            sequence=42,
            utime_us=123_456,
            joint_position_rad=tuple(0.1 * index for index in range(7)),
            joint_velocity_rad_s=tuple(-0.2 * index for index in range(7)),
            joint_effort_nm=tuple(float(index) for index in range(7)),
            object_quaternion_wxyz=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
            object_position_m=(0.4, 0.2, -0.016),
            object_angular_velocity_rad_s=(0.0, 0.0, 0.3),
            object_linear_velocity_m_s=(0.01, -0.02, 0.0),
        )
        packet = state.pack()
        self.assertEqual(len(packet), STATE_PACKET_SIZE)
        self.assertEqual(C3MeasuredState.unpack(packet), state)

    def test_safe_contact_state_flag_round_trip(self) -> None:
        state = C3MeasuredState(
            sequence=1,
            utime_us=2,
            joint_position_rad=(0.0,) * 7,
            joint_velocity_rad_s=(0.0,) * 7,
            joint_effort_nm=(0.0,) * 7,
            object_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            object_position_m=(0.4, 0.2, 0.01),
            object_angular_velocity_rad_s=(0.0,) * 3,
            object_linear_velocity_m_s=(0.0,) * 3,
            flags=(
                C3StateFlags.LEGAL_SAFE_CONTACT | C3StateFlags.FORCE_C3_MODE
            ),
        )
        decoded = C3MeasuredState.unpack(state.pack())
        self.assertEqual(
            decoded.flags,
            C3StateFlags.LEGAL_SAFE_CONTACT | C3StateFlags.FORCE_C3_MODE,
        )

    def test_command_round_trip_and_status_bits(self) -> None:
        command = C3EffortCommand(
            sequence=9,
            utime_us=1000,
            flags=C3CommandFlags.READY | C3CommandFlags.SEMANTIC_HOLD,
            joint_effort_nm=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        )
        packet = command.pack()
        self.assertEqual(len(packet), COMMAND_PACKET_SIZE)
        decoded = C3EffortCommand.unpack(packet)
        self.assertEqual(decoded, command)
        self.assertTrue(decoded.flags & C3CommandFlags.SEMANTIC_HOLD)

    def test_scene_state_round_trip_has_fixed_size(self) -> None:
        state = C3MeasuredSceneState(
            sequence=43,
            utime_us=223_456,
            joint_position_rad=(0.1,) * 7,
            joint_velocity_rad_s=(0.2,) * 7,
            joint_effort_nm=(0.3,) * 7,
            target=self.rigid_body(0.4),
            clutter=(self.rigid_body(0.5), self.rigid_body(0.6)),
        )
        packet = state.pack()
        self.assertEqual(len(packet), SCENE_STATE_PACKET_SIZE)
        self.assertEqual(C3MeasuredSceneState.unpack(packet), state)

    def test_scene_state_rejects_excess_or_invalid_clutter(self) -> None:
        state = C3MeasuredSceneState(
            sequence=0,
            utime_us=0,
            joint_position_rad=(0.0,) * 7,
            joint_velocity_rad_s=(0.0,) * 7,
            joint_effort_nm=(0.0,) * 7,
            target=self.rigid_body(0.4),
            clutter=tuple(
                self.rigid_body(0.5 + index)
                for index in range(MAX_CLUTTER_OBJECTS + 1)
            ),
        )
        with self.assertRaisesRegex(ValueError, "at most"):
            state.pack()

        invalid = C3MeasuredSceneState(
            sequence=0,
            utime_us=0,
            joint_position_rad=(0.0,) * 7,
            joint_velocity_rad_s=(0.0,) * 7,
            joint_effort_nm=(0.0,) * 7,
            target=self.rigid_body(0.4),
            clutter=(
                C3RigidBodyState(
                    quaternion_wxyz=(2.0, 0.0, 0.0, 0.0),
                    position_m=(0.5, 0.2, 0.01),
                    angular_velocity_rad_s=(0.0,) * 3,
                    linear_velocity_m_s=(0.0,) * 3,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "normalized"):
            invalid.pack()

    def test_scene_state_rejects_noncanonical_network_framing(self) -> None:
        state = C3MeasuredSceneState(
            sequence=1,
            utime_us=2,
            joint_position_rad=(0.0,) * 7,
            joint_velocity_rad_s=(0.0,) * 7,
            joint_effort_nm=(0.0,) * 7,
            target=self.rigid_body(0.4),
        )
        bad_count = bytearray(state.pack())
        struct.pack_into("!H", bad_count, struct.calcsize("!4sHHQq"), 4)
        with self.assertRaisesRegex(ValueError, "clutter count"):
            C3MeasuredSceneState.unpack(bytes(bad_count))

        bad_padding = bytearray(state.pack())
        header_size = struct.calcsize("!4sHHQqH")
        first_inactive_slot = header_size + (21 + 13) * struct.calcsize("!d")
        struct.pack_into("!d", bad_padding, first_inactive_slot, 0.5)
        with self.assertRaisesRegex(ValueError, "canonical padding"):
            C3MeasuredSceneState.unpack(bytes(bad_padding))

    def test_rejects_non_normalized_object_quaternion(self) -> None:
        state = C3MeasuredState(
            sequence=0,
            utime_us=0,
            joint_position_rad=(0.0,) * 7,
            joint_velocity_rad_s=(0.0,) * 7,
            joint_effort_nm=(0.0,) * 7,
            object_quaternion_wxyz=(2.0, 0.0, 0.0, 0.0),
            object_position_m=(0.0, 0.0, 0.0),
            object_angular_velocity_rad_s=(0.0, 0.0, 0.0),
            object_linear_velocity_m_s=(0.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(ValueError, "normalized"):
            state.pack()

    def test_rejects_bad_magic_and_unknown_command_flags(self) -> None:
        valid = C3EffortCommand(
            sequence=0,
            utime_us=0,
            flags=C3CommandFlags.READY,
            joint_effort_nm=(0.0,) * 7,
        ).pack()
        with self.assertRaisesRegex(ValueError, "header"):
            C3EffortCommand.unpack(b"BAD!" + valid[4:])

        mutable = bytearray(valid)
        # Header is network-endian: magic[4], version uint16, flags uint16.
        struct.pack_into("!H", mutable, 6, 1 << 15)
        with self.assertRaisesRegex(ValueError, "unknown status"):
            C3EffortCommand.unpack(bytes(mutable))

    def test_rejects_truncated_packet(self) -> None:
        with self.assertRaisesRegex(ValueError, "bytes"):
            C3MeasuredState.unpack(b"short")

    def test_rejects_nonfinite_network_state(self) -> None:
        state = C3MeasuredState(
            sequence=0,
            utime_us=0,
            joint_position_rad=(0.0,) * 7,
            joint_velocity_rad_s=(0.0,) * 7,
            joint_effort_nm=(0.0,) * 7,
            object_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            object_position_m=(0.0, 0.0, 0.0),
            object_angular_velocity_rad_s=(0.0, 0.0, 0.0),
            object_linear_velocity_m_s=(0.0, 0.0, 0.0),
        )
        packet = bytearray(state.pack())
        # First double starts after magic/version/flags/sequence/utime.
        struct.pack_into("!d", packet, struct.calcsize("!4sHHQq"), math.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            C3MeasuredState.unpack(bytes(packet))

    def test_task_command_round_trip(self) -> None:
        command = C3TaskCommand(
            sequence=9,
            utime_us=345_000,
            flags=(
                C3CommandFlags.READY
                | C3CommandFlags.C3_MODE
                | C3CommandFlags.YAW_RECOVERY
            ),
            position_m=(0.41, 0.22, 0.08),
            velocity_m_s=(0.02, -0.01, 0.0),
            feedforward_force_n=(1.0, 2.0, 3.0),
        )
        packet = command.pack()
        self.assertEqual(len(packet), TASK_COMMAND_PACKET_SIZE)
        self.assertEqual(C3TaskCommand.unpack(packet), command)
        self.assertTrue(
            C3TaskCommand.unpack(packet).flags & C3CommandFlags.C3_MODE
        )
        self.assertTrue(
            C3TaskCommand.unpack(packet).flags & C3CommandFlags.YAW_RECOVERY
        )

    def test_task_command_rejects_nonfinite_value(self) -> None:
        command = C3TaskCommand(
            sequence=9,
            utime_us=345_000,
            flags=C3CommandFlags.READY,
            position_m=(math.nan, 0.22, 0.08),
            velocity_m_s=(0.0, 0.0, 0.0),
            feedforward_force_n=(0.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            command.pack()


if __name__ == "__main__":
    unittest.main()
