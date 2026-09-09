#!/usr/bin/env python3
"""Send deterministic fake states through the native C3+ TCP relay."""

from __future__ import annotations

import argparse
import math
import socket
import time

from dapl.contact_planner.c3_online_protocol import (
    C3CommandFlags,
    C3EffortCommand,
    C3MeasuredState,
    COMMAND_PACKET_SIZE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7790)
    parser.add_argument("--packets", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--period-s", type=float, default=0.0)
    return parser.parse_args()


def receive_exact(connection: socket.socket, size: int) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        chunk = connection.recv(size - len(payload))
        if not chunk:
            raise ConnectionError("relay closed inside a command packet")
        payload.extend(chunk)
    return bytes(payload)


def main() -> int:
    args = parse_args()
    if args.packets <= 0 or args.timeout_s <= 0.0 or args.period_s < 0.0:
        raise ValueError("packet count/timeout must be positive and period non-negative")
    initial_q = (2.191, 1.1, -1.33, -2.22, 1.30, 2.02, 0.08)
    commands: list[C3EffortCommand] = []
    with socket.create_connection((args.host, args.port), timeout=args.timeout_s) as client:
        client.settimeout(args.timeout_s)
        for sequence in range(args.packets):
            state = C3MeasuredState(
                sequence=sequence,
                utime_us=100_000 + sequence * max(1, round(args.period_s * 1.0e6)),
                joint_position_rad=initial_q,
                joint_velocity_rad_s=(0.0,) * 7,
                joint_effort_nm=(0.0,) * 7,
                object_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                object_position_m=(0.4, 0.2, -0.016),
                object_angular_velocity_rad_s=(0.0, 0.0, 0.0),
                object_linear_velocity_m_s=(0.0, 0.0, 0.0),
            )
            client.sendall(state.pack())
            command = C3EffortCommand.unpack(
                receive_exact(client, COMMAND_PACKET_SIZE)
            )
            if command.sequence != sequence:
                raise RuntimeError(
                    f"relay sequence mismatch: {command.sequence} != {sequence}"
                )
            commands.append(command)
            if args.period_s:
                time.sleep(args.period_s)
    if not commands:
        raise RuntimeError("relay returned no commands")
    print(
        "C3_ONLINE_RELAY_SMOKE_PASS",
        f"packets={len(commands)}",
        f"ready={sum(bool(item.flags & C3CommandFlags.READY) for item in commands)}",
        f"stale={sum(bool(item.flags & C3CommandFlags.STALE_STATE) for item in commands)}",
        f"fresh={sum(bool(item.flags & C3CommandFlags.READY) and not bool(item.flags & C3CommandFlags.STALE_STATE) for item in commands)}",
        f"max_effort_l2_nm={max(math.sqrt(sum(value * value for value in item.joint_effort_nm)) for item in commands):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
