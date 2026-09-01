#!/usr/bin/env python3
"""Minimal local TCPQ broker for LCM clients.

This implements LCM's documented TCPQ framing and keeps all traffic on the
loopback interface.  It is useful on hosts where UDP multicast is disabled.
"""

from __future__ import annotations

import argparse
import re
import socket
import socketserver
import struct
import threading
from typing import Optional


MAGIC_SERVER = 0x287617FA
MAGIC_CLIENT = 0x287617FB
PROTOCOL_VERSION = 0x0100
MESSAGE_TYPE_PUBLISH = 1
MESSAGE_TYPE_SUBSCRIBE = 2
MESSAGE_TYPE_UNSUBSCRIBE = 3
MAX_CHANNEL_BYTES = 1 << 16
MAX_MESSAGE_BYTES = 1 << 28
UINT32 = struct.Struct("!I")


def recv_exact(sock: socket.socket, length: int) -> Optional[bytes]:
    chunks = bytearray()
    while len(chunks) < length:
        try:
            chunk = sock.recv(length - len(chunks))
        except ConnectionError:
            return None
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def recv_u32(sock: socket.socket) -> Optional[int]:
    data = recv_exact(sock, UINT32.size)
    return None if data is None else UINT32.unpack(data)[0]


class TcpqServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address):
        super().__init__(server_address, TcpqHandler)
        self.clients: set[TcpqHandler] = set()
        self.clients_lock = threading.Lock()

    def add_client(self, client: "TcpqHandler") -> None:
        with self.clients_lock:
            self.clients.add(client)

    def remove_client(self, client: "TcpqHandler") -> None:
        with self.clients_lock:
            self.clients.discard(client)

    def relay(self, channel: bytes, data: bytes) -> None:
        channel_text = channel.decode("ascii")
        with self.clients_lock:
            clients = tuple(self.clients)
        for client in clients:
            client.send_if_subscribed(channel_text, channel, data)


class TcpqHandler(socketserver.BaseRequestHandler):
    server: TcpqServer

    def setup(self) -> None:
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.request.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        self.request.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.subscriptions: dict[str, re.Pattern[str]] = {}
        self.subscription_lock = threading.Lock()
        self.send_lock = threading.Lock()

    def handle(self) -> None:
        client_magic = recv_u32(self.request)
        client_version = recv_u32(self.request)
        if client_magic != MAGIC_CLIENT or client_version is None:
            return
        self.request.sendall(UINT32.pack(MAGIC_SERVER) + UINT32.pack(PROTOCOL_VERSION))
        self.server.add_client(self)
        try:
            while True:
                message_type = recv_u32(self.request)
                if message_type is None:
                    return
                if message_type == MESSAGE_TYPE_PUBLISH:
                    channel = self._recv_sized(MAX_CHANNEL_BYTES)
                    data = self._recv_sized(MAX_MESSAGE_BYTES)
                    if channel is None or data is None:
                        return
                    self.server.relay(channel, data)
                elif message_type in (MESSAGE_TYPE_SUBSCRIBE, MESSAGE_TYPE_UNSUBSCRIBE):
                    channel = self._recv_sized(MAX_CHANNEL_BYTES)
                    if channel is None:
                        return
                    channel_text = channel.decode("ascii")
                    with self.subscription_lock:
                        if message_type == MESSAGE_TYPE_SUBSCRIBE:
                            self.subscriptions[channel_text] = re.compile(channel_text)
                        else:
                            self.subscriptions.pop(channel_text, None)
                else:
                    return
        finally:
            self.server.remove_client(self)

    def _recv_sized(self, maximum: int) -> Optional[bytes]:
        length = recv_u32(self.request)
        if length is None or length > maximum:
            return None
        return recv_exact(self.request, length)

    def send_if_subscribed(self, channel_text: str, channel: bytes, data: bytes) -> None:
        with self.subscription_lock:
            matches = any(
                pattern.fullmatch(channel_text)
                for pattern in self.subscriptions.values()
            )
        if not matches:
            return
        frame = b"".join(
            (
                UINT32.pack(MESSAGE_TYPE_PUBLISH),
                UINT32.pack(len(channel)),
                channel,
                UINT32.pack(len(data)),
                data,
            )
        )
        try:
            with self.send_lock:
                self.request.sendall(frame)
        except OSError:
            self.server.remove_client(self)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7700)
    args = parser.parse_args()
    with TcpqServer((args.host, args.port)) as server:
        print(f"LCM TCPQ hub listening on {args.host}:{args.port}", flush=True)
        server.serve_forever(poll_interval=0.2)


if __name__ == "__main__":
    main()
