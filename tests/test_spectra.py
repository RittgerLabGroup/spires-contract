import numpy as np
import pytest
import xarray as xr

from spires_contract import spectra
from spires_contract._validate import ContractError


def make_target(dims=("y", "x", "band"), dtype=np.float32, with_band_coord=True):
    shape = tuple({"y": 3, "x": 4, "band": 5}[d] for d in dims)
    coords = {"band": np.arange(shape[dims.index("band")])} if with_band_coord and "band" in dims else {}
    return xr.DataArray(np.zeros(shape, dtype=dtype), dims=dims, coords=coords)


def make_solar(dims=("y", "x"), dtype=np.float32):
    sizes = {"y": 3, "x": 4}
    shape = tuple(sizes[d] for d in dims)
    return xr.DataArray(
        np.zeros(shape, dtype=dtype),
        dims=dims,
        attrs={"units": "degrees"},
    )


def test_validate_target_spectra_accepts_valid():
    spectra.validate_target_spectra(make_target())  # must not raise


def test_validate_target_spectra_rejects_missing_band_dim():
    da = make_target(dims=("y", "x"), with_band_coord=False)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    assert "band" in str(exc.value)


def test_validate_target_spectra_accepts_float32():
    # float32 is the canonical boundary dtype.
    spectra.validate_target_spectra(make_target(dtype=np.float32))  # must not raise


def test_validate_target_spectra_rejects_float64():
    # float64 is no longer accepted: the boundary is float32-only so the batch
    # inversion path stays a single deterministic kernel.
    da = make_target(dtype=np.float64)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    assert "float64" in str(exc.value)


def test_validate_solar_angles_accepts_float32():
    spectra.validate_solar_angles(make_solar(dtype=np.float32))  # must not raise


def test_validate_target_spectra_rejects_non_float_dtype():
    # A non-float dtype (e.g. int) is still a violation.
    da = make_target(dtype=np.int32)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    text = str(exc.value)
    assert "int32" in text and "float32" in text  # message names actual + expected dtype


def test_validate_target_spectra_rejects_missing_band_coord():
    da = make_target(with_band_coord=False)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    assert "band" in str(exc.value)


def test_validate_target_spectra_collects_multiple_violations():
    # wrong dtype AND missing band coordinate -> both reported in one error
    da = make_target(dtype=np.int32, with_band_coord=False)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    text = str(exc.value)
    assert "int32" in text
    assert "band" in text


def test_validate_target_spectra_rejects_wrong_dim_order():
    # order is part of the contract: a transposed array must be rejected
    da = make_target(dims=("band", "y", "x"))
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    assert "canonical order" in str(exc.value)


def test_validate_solar_angles_rejects_wrong_dim_order():
    da = make_solar(dims=("x", "y"))
    with pytest.raises(ContractError) as exc:
        spectra.validate_solar_angles(da)
    assert "canonical order" in str(exc.value)


def test_validate_background_spectra_accepts_valid():
    spectra.validate_background_spectra(make_target())  # same shape rules as target


def test_validate_solar_angles_accepts_valid():
    spectra.validate_solar_angles(make_solar())


def test_validate_solar_angles_rejects_band_dim():
    da = xr.DataArray(np.zeros((3, 4, 5), dtype=np.float64), dims=("y", "x", "band"))
    with pytest.raises(ContractError):
        spectra.validate_solar_angles(da)


def test_validate_target_spectra_rejects_extra_dim():
    da = make_target(dims=("y", "x", "band"))
    da = da.expand_dims("time")  # now (time, y, x, band)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    assert "time" in str(exc.value)


def test_validate_solar_angles_rejects_extra_dim():
    da = make_solar().expand_dims("time")  # (time, y, x)
    with pytest.raises(ContractError) as exc:
        spectra.validate_solar_angles(da)
    assert "time" in str(exc.value)
