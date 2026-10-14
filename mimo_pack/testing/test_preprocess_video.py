"""Tests for mimo_pack.preprocess.video."""

import cv2
import numpy as np
import pandas as pd
import pytest

from mimo_pack.preprocess.video import tif_to_video


def _count_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Could not open video file: {video_path}"

    frame_count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        frame_count += 1

    cap.release()
    return frame_count


def test_tif_to_video_writes_avi_with_all_frames(tmp_path):
    frames = [0, 85, 170]
    names = [
        "reach_0001_0000.tif",
        "reach_0001_0001.tif",
        "reach_0002_0000.tif",
    ]

    for name, value in zip(names, frames):
        img = np.full((24, 32), value, dtype=np.uint8)
        ok = cv2.imwrite(str(tmp_path / name), img)
        assert ok, f"Failed to write synthetic TIF {name}"

    vid_path, tbl_path = tif_to_video(str(tmp_path), fps=10)

    assert vid_path.endswith(".avi")
    assert tbl_path.endswith(".csv")
    assert (tmp_path / "reach.avi").exists()
    assert _count_video_frames(vid_path) == len(names)

    tbl_df = pd.read_csv(tbl_path, index_col=0)
    assert len(tbl_df) == len(names)
    assert list(tbl_df["tif_file"]) == names
    assert list(tbl_df["epoch_num"]) == [1, 1, 2]
    assert list(tbl_df["frame_num"]) == [0, 1, 0]


def test_tif_to_video_raises_on_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="No TIF files found"):
        tif_to_video(str(tmp_path))


if __name__ == "__main__":
    pytest.main([__file__])