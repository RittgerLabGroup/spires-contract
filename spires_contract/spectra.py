"""I/O -> inversion boundary contract: target/background spectra + solar angles.

Canonical forms (see conventions):
- target/background spectra: dims (y, x, band), float32, with a `band` coordinate
- solar zenith:              dims (y, x),       float32 physical degrees,
  without active scale/offset packing metadata

Dimension ORDER is part of the contract — the C++ inversion kernel indexes
arrays positionally, so a transposed array is as wrong as a missing dimension.
`validate_*` raises ContractError listing every violation (missing/extra dims,
wrong order, wrong dtype, missing coordinate) so a producer gets one actionable
error.
"""

import numpy as np

__all__ = [
    "validate_target_spectra",
    "validate_background_spectra",
    "validate_solar_angles",
]

from spires_contract import conventions as c
from spires_contract._validate import (
    check_coords_present,
    check_dims_present,
    check_dims_order,
    check_dtype,
    check_no_extra_dims,
    raise_if_violations,
)


def _validate_spectra(da, contract_name):
    violations = []
    violations += check_dims_present(da, c.SPECTRA_DIMS)
    violations += check_no_extra_dims(da, c.SPECTRA_DIMS)
    violations += check_dims_order(da, c.SPECTRA_DIMS)
    violations += check_dtype(da, c.ACCEPTED_DTYPES)
    violations += check_coords_present(da, ("band",))
    raise_if_violations(contract_name, violations)


def validate_target_spectra(da):
    """Validate mixed target reflectance spectra. Raises ContractError."""
    _validate_spectra(da, "target_spectra")


def validate_background_spectra(da):
    """Validate background (R_0) reflectance spectra. Raises ContractError."""
    _validate_spectra(da, "background_spectra")


def validate_solar_angles(da):
    """Validate unpacked per-pixel solar zenith angles. Raises ContractError."""
    violations = []
    violations += check_dims_present(da, c.SOLAR_ANGLE_DIMS)
    violations += check_no_extra_dims(da, c.SOLAR_ANGLE_DIMS)
    violations += check_dims_order(da, c.SOLAR_ANGLE_DIMS)
    violations += check_dtype(da, c.ACCEPTED_DTYPES)

    units = da.attrs.get("units")
    normalized_units = units.strip().lower() if isinstance(units, str) else None
    if normalized_units not in c.SOLAR_ANGLE_UNIT_ALIASES:
        violations.append(
            "solar zenith units must identify physical degrees "
            f"(accepted: {c.SOLAR_ANGLE_UNIT_ALIASES}; got {units!r})"
        )

    for key in ("scale_factor", "add_offset"):
        locations = []
        if da.attrs.get(key) is not None:
            locations.append("attrs")
        if da.encoding.get(key) is not None:
            locations.append("encoding")
        if locations:
            violations.append(
                f"solar zenith must be unpacked physical values; found active "
                f"{key!r} metadata in {' and '.join(locations)}"
            )

    if np.issubdtype(da.dtype, np.floating):
        values = np.asarray(da.values)
        if np.isinf(values).any():
            violations.append("solar zenith contains infinite values")

        finite_values = values[np.isfinite(values)]
        minimum, maximum = c.SOLAR_ZENITH_RANGE_DEGREES
        if finite_values.size and (
            (finite_values < minimum).any() or (finite_values > maximum).any()
        ):
            violations.append(
                "finite solar zenith values must be within "
                f"[{minimum:g}, {maximum:g}] degrees "
                f"(observed [{finite_values.min():g}, {finite_values.max():g}])"
            )

    # solar angles are 2-D: a band dimension is a violation
    if "band" in da.dims:
        violations.append(
            f"unexpected dimension 'band' for solar angles (dims: {tuple(da.dims)})"
        )
    raise_if_violations("solar_angles", violations)
