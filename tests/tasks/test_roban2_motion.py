"""OpenLET trajectory schema and ordering checks."""

import numpy as np
import pytest

from unisupermimic.tasks.motion_tracking.roban2.supermimic.motion import _load_npz


@pytest.mark.parametrize("width", [28, 30])
def test_load_openlet_qpos(tmp_path, width):
    path = tmp_path / "clip.npz"
    values = np.zeros((3, width), dtype=np.float32)
    values[:, 2] = 0.77
    values[:, 3] = 1.0
    values[:, 7:] = np.arange(width - 7, dtype=np.float32)
    np.savez(path, qpos=values, fps=120.0)

    result = _load_npz(str(path), 120.0)

    assert result.shape == (3, 30)
    np.testing.assert_array_equal(result[:, :width], values)
    if width == 28:
        np.testing.assert_array_equal(result[:, 28:], 0)


def test_load_openlet_data_and_reject_bad_metadata(tmp_path):
    path = tmp_path / "clip.npz"
    values = np.zeros((2, 30), dtype=np.float32)
    values[:, 3] = 1.0
    np.savez(path, data=values, fps=120.0)
    assert _load_npz(str(path), 120.0).shape == (2, 30)

    with pytest.raises(ValueError, match="fps"):
        _load_npz(str(path), 30.0)
    values[0, 3] = 0.0
    np.savez(path, data=values, fps=120.0)
    with pytest.raises(ValueError, match="quaternions"):
        _load_npz(str(path), 120.0)
