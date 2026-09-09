"""Bounded-memory RGB video recording for long simulator rollouts."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


class StreamingRgbVideoWriter:
    """Stream RGB frames to FFmpeg instead of retaining a rollout in RAM."""

    def __init__(self, path: Path, *, fps: float = 20.0, crf: int = 20):
        if fps <= 0.0:
            raise ValueError("video fps must be positive")
        if not 0 <= crf <= 51:
            raise ValueError("video CRF must be between 0 and 51")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError("ffmpeg is required for streaming video")
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.crf = int(crf)
        self._ffmpeg = ffmpeg
        self._process: subprocess.Popen | None = None
        self._size: tuple[int, int] | None = None
        self.frames_written = 0

    @staticmethod
    def _rgb_frame(frame) -> np.ndarray:
        rgb = np.asarray(frame)
        if rgb.ndim == 4 and rgb.shape[0] == 1:
            rgb = rgb[0]
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError(f"expected HxWx3/4 RGB frame, got {rgb.shape}")
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)

    def _start(self, rgb: np.ndarray) -> None:
        height, width = rgb.shape[:2]
        self._size = (width, height)
        self._process = subprocess.Popen(
            [
                self._ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                f"{self.fps:g}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                str(self.crf),
                "-pix_fmt",
                "yuv420p",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame) -> None:
        rgb = self._rgb_frame(frame)
        if self._process is None:
            self._start(rgb)
        if self._size != (rgb.shape[1], rgb.shape[0]):
            raise ValueError(
                f"video frame size changed from {self._size} "
                f"to {(rgb.shape[1], rgb.shape[0])}"
            )
        assert self._process is not None and self._process.stdin is not None
        try:
            self._process.stdin.write(rgb.tobytes())
        except BrokenPipeError as exc:
            stderr = (
                self._process.stderr.read().decode("utf-8", errors="replace")
                if self._process.stderr is not None
                else ""
            )
            raise RuntimeError(f"ffmpeg video pipe failed: {stderr.strip()}") from exc
        self.frames_written += 1

    def close(self) -> dict[str, object]:
        if self._process is None:
            return {
                "path": str(self.path),
                "fps": self.fps,
                "frames": 0,
                "duration_s": 0.0,
                "encoded": False,
            }
        assert self._process.stdin is not None
        self._process.stdin.close()
        return_code = self._process.wait(timeout=60.0)
        stderr = (
            self._process.stderr.read().decode("utf-8", errors="replace")
            if self._process.stderr is not None
            else ""
        )
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg exited with status {return_code}: {stderr.strip()}"
            )
        return {
            "path": str(self.path),
            "fps": self.fps,
            "frames": self.frames_written,
            "duration_s": self.frames_written / self.fps,
            "encoded": self.path.is_file() and self.path.stat().st_size > 0,
            "width": self._size[0],
            "height": self._size[1],
        }
