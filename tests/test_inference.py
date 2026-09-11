import numpy as np
import pytest

from obsea_swh.inference import _as_window, _canonical_columns


def test_window_validation_accepts_expected_shape():
    window = _as_window(np.zeros((16, 4)), look_back=16, n_features=4)
    assert window.shape == (16, 4)
    assert window.dtype == np.float32


def test_window_validation_rejects_wrong_shape():
    with pytest.raises(ValueError, match="Expected input shape"):
        _as_window(np.zeros((15, 4)), look_back=16, n_features=4)


def test_window_validation_rejects_non_finite_values():
    values = np.zeros((16, 4))
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        _as_window(values, look_back=16, n_features=4)


def test_csv_alias_is_supported():
    columns = _canonical_columns(["VMDR", "VTPK", "VZMX", "VTZA", "VHM0"])
    assert columns["VDMR"] == "VMDR"
