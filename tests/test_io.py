import numpy as np
import pytest

from signal_lab.io import save_iq_npy, load_iq_npy


def test_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(968)
    iq = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex128)

    path = tmp_path / "iq_test.npy"
    saved_path = save_iq_npy(path, iq)
    iq2 = load_iq_npy(saved_path)

    assert iq2.shape == iq.shape
    assert np.issubdtype(iq2.dtype, np.complexfloating)
    assert np.allclose(iq2, iq)


def test_save_adds_suffix_if_missing(tmp_path):
    iq = np.ones(8, dtype=np.complex128)
    path = tmp_path / "iq_no_suffix"
    saved_path = save_iq_npy(path, iq)
    assert saved_path.suffix == ".npy"
    assert saved_path.exists()


def test_save_rejects_real(tmp_path):
    iq = np.ones(8, dtype=np.float64)
    with pytest.raises(ValueError):
        save_iq_npy(tmp_path / "x.npy", iq)


def test_save_rejects_2d(tmp_path):
    iq = np.ones((2, 4), dtype=np.complex128)
    with pytest.raises(ValueError):
        save_iq_npy(tmp_path / "x.npy", iq)


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_iq_npy(tmp_path / "missing.npy")


def test_load_rejects_non_complex_file(tmp_path):
    path = tmp_path / "real.npy"
    np.save(path, np.ones(8, dtype=np.float64))
    with pytest.raises(ValueError):
        load_iq_npy(path)
