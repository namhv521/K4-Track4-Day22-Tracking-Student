"""Kiểm tra số thứ tự frame bằng ảnh tổng hợp, không cần trọng số."""

import cv2
import numpy as np

from run_tracking import iter_frames


def test_numbered_images_keep_original_frame_indices(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for name in ("000001.jpg", "000003.jpg"):
        assert cv2.imwrite(str(tmp_path / name), image)
    assert [index for index, _ in iter_frames(tmp_path)] == [0, 2]
