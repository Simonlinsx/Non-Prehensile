from __future__ import annotations

import shutil

import numpy as np
import pytest

from dapl.contact_planner.streaming_video import StreamingRgbVideoWriter


def test_streaming_video_rejects_invalid_rate(tmp_path):
    with pytest.raises(ValueError, match="fps"):
        StreamingRgbVideoWriter(tmp_path / "invalid.mp4", fps=0.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_streaming_video_writes_bounded_frame_sequence(tmp_path):
    path = tmp_path / "smoke.mp4"
    writer = StreamingRgbVideoWriter(path, fps=10.0)
    for index in range(4):
        writer.write(np.full((48, 64, 3), 40 * index, dtype=np.uint8))
    report = writer.close()

    assert report["encoded"] is True
    assert report["frames"] == 4
    assert report["duration_s"] == pytest.approx(0.4)
    assert report["width"] == 64
    assert report["height"] == 48
    assert path.stat().st_size > 0
