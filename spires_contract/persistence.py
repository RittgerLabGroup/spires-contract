"""Versioned scientific conventions for persisted ``SpiresData`` products.

This module defines scientific file content independently of filesystem policy.
``spires-io`` owns NetCDF mechanics, atomic promotion, and physical encodings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from spires_contract import conventions as c
from spires_contract._validate import (
    ContractError,
    check_coords_match,
    check_coords_present,
    check_dims_order,
    check_dims_present,
    check_dtype,
    check_no_extra_dims,
    raise_if_violations,
)
from spires_contract.data import SpiresData, validate_spires_data
from spires_contract.inversion_input import validate_for_inversion
from spires_contract.results import validate_results

__all__ = [
    "ALBEDO_RESULT_VARIABLES",
    "COMPLETION_STATUS_COMPLETE",
    "CONTENT_PROFILE_INVERSION_RAW",
    "CONTENT_PROFILE_POSTPROCESSED_RAW",
    "DERIVED_RESULT_LONG_NAMES",
    "DERIVED_RESULT_UNITS",
    "OPERATION_ALBEDO",
    "OPERATION_CANOPY_CORRECTION",
    "OPERATION_DELTA_VIS",
    "OPERATION_ICE_ADJUSTMENT",
    "OPERATION_RADIATIVE_FORCING",
    "OPERATION_RESULT_VARIABLES",
    "PERSISTED_ATTR_ACQUISITION_TIME",
    "PERSISTED_ATTR_BACKGROUND_VARIABLE",
    "PERSISTED_ATTR_COMPLETED_OPERATIONS",
    "PERSISTED_ATTR_COMPLETED_STAGES",
    "PERSISTED_ATTR_COMPLETION_STATUS",
    "PERSISTED_ATTR_CONTENT_PROFILE",
    "PERSISTED_ATTR_CREATED_AT",
    "PERSISTED_ATTR_GRID_DIGEST",
    "PERSISTED_ATTR_PACKAGE_VERSIONS",
    "PERSISTED_ATTR_PLATFORM",
    "PERSISTED_ATTR_PRESENT_GROUPS",
    "PERSISTED_ATTR_PRODUCT",
    "PERSISTED_ATTR_PRODUCT_CONTENTS",
    "PERSISTED_ATTR_PRODUCT_TYPE",
    "PERSISTED_ATTR_PROVENANCE",
    "PERSISTED_ATTR_SCHEMA_VERSION",
    "PERSISTED_ATTR_SENSOR",
    "PERSISTED_ATTR_SPATIAL_ID",
    "PERSISTED_ATTR_UPDATED_AT",
    "PERSISTED_GROUP_ANCILLARY",
    "PERSISTED_GROUP_BACKGROUND",
    "PERSISTED_GROUP_RESULTS",
    "PERSISTED_GROUP_SCENE",
    "PERSISTED_GROUPS",
    "PERSISTED_PRODUCT_TYPE",
    "PERSISTED_SCHEMA_VERSION",
    "PRODUCT_CONTENTS_FULL",
    "PRODUCT_CONTENTS_RESULTS_SUBSET",
    "ProductIdentity",
    "PersistedProductMetadata",
    "validate_persisted_data",
    "validate_persisted_grid",
    "validate_persisted_metadata",
]


PERSISTED_SCHEMA_VERSION = 1
PERSISTED_PRODUCT_TYPE = "SpiresData"

PERSISTED_GROUP_SCENE = "scene"
PERSISTED_GROUP_BACKGROUND = "background"
PERSISTED_GROUP_ANCILLARY = "ancillary"
PERSISTED_GROUP_RESULTS = "results"
PERSISTED_GROUPS = (
    PERSISTED_GROUP_SCENE,
    PERSISTED_GROUP_BACKGROUND,
    PERSISTED_GROUP_ANCILLARY,
    PERSISTED_GROUP_RESULTS,
)

PERSISTED_ATTR_PRODUCT_TYPE = "spires_product_type"
PERSISTED_ATTR_SCHEMA_VERSION = "spires_storage_schema_version"
PERSISTED_ATTR_PRESENT_GROUPS = "spires_present_groups"
PERSISTED_ATTR_BACKGROUND_VARIABLE = "spires_background_variable"
PERSISTED_ATTR_CONTENT_PROFILE = "spires_content_profile"
PERSISTED_ATTR_PRODUCT_CONTENTS = "spires_product_contents"
PERSISTED_ATTR_COMPLETION_STATUS = "spires_completion_status"
PERSISTED_ATTR_COMPLETED_STAGES = "spires_completed_stages"
PERSISTED_ATTR_COMPLETED_OPERATIONS = "spires_completed_operations"
PERSISTED_ATTR_SENSOR = "spires_sensor"
PERSISTED_ATTR_PRODUCT = "spires_source_product"
PERSISTED_ATTR_PLATFORM = "spires_platform"
PERSISTED_ATTR_SPATIAL_ID = "spires_spatial_id"
PERSISTED_ATTR_ACQUISITION_TIME = "spires_acquisition_time"
PERSISTED_ATTR_GRID_DIGEST = "spires_grid_digest"
PERSISTED_ATTR_CREATED_AT = "spires_created_at"
PERSISTED_ATTR_UPDATED_AT = "spires_updated_at"
PERSISTED_ATTR_PACKAGE_VERSIONS = "spires_package_versions"
PERSISTED_ATTR_PROVENANCE = "spires_provenance"

CONTENT_PROFILE_INVERSION_RAW = "inversion_raw"
CONTENT_PROFILE_POSTPROCESSED_RAW = "postprocessed_raw"
CONTENT_PROFILES = (
    CONTENT_PROFILE_INVERSION_RAW,
    CONTENT_PROFILE_POSTPROCESSED_RAW,
)

PRODUCT_CONTENTS_FULL = "full"
PRODUCT_CONTENTS_RESULTS_SUBSET = "results_subset"
PRODUCT_CONTENTS = (
    PRODUCT_CONTENTS_FULL,
    PRODUCT_CONTENTS_RESULTS_SUBSET,
)

COMPLETION_STATUS_COMPLETE = "complete"

OPERATION_CANOPY_CORRECTION = "canopy_correction"
OPERATION_ICE_ADJUSTMENT = "ice_adjustment"
OPERATION_ALBEDO = "albedo"
OPERATION_DELTA_VIS = "delta_vis"
OPERATION_RADIATIVE_FORCING = "radiative_forcing"
COMPLETED_OPERATION_ORDER = (
    OPERATION_CANOPY_CORRECTION,
    OPERATION_ICE_ADJUSTMENT,
    OPERATION_ALBEDO,
    OPERATION_DELTA_VIS,
    OPERATION_RADIATIVE_FORCING,
)

ALBEDO_RESULT_VARIABLES = (
    "albedo_clean_flat",
    "albedo_dirty_flat",
    "albedo_clean_terrain_corrected",
    "albedo_dirty_terrain_corrected",
)
OPERATION_RESULT_VARIABLES = {
    OPERATION_CANOPY_CORRECTION: ("canopy_adjusted_fsnow",),
    OPERATION_ICE_ADJUSTMENT: ("ice_adjusted_fsnow",),
    OPERATION_ALBEDO: ALBEDO_RESULT_VARIABLES,
    OPERATION_DELTA_VIS: ("delta_vis",),
    OPERATION_RADIATIVE_FORCING: ("radiative_forcing",),
}

DERIVED_RESULT_UNITS = {
    "canopy_adjusted_fsnow": "1",
    "ice_adjusted_fsnow": "1",
    "albedo_clean_flat": "1",
    "albedo_dirty_flat": "1",
    "albedo_clean_terrain_corrected": "1",
    "albedo_dirty_terrain_corrected": "1",
    "delta_vis": "1",
    "radiative_forcing": "W m-2",
}
DERIVED_RESULT_LONG_NAMES = {
    "canopy_adjusted_fsnow": (
        "Snow fraction adjusted for shade and canopy obstruction"
    ),
    "ice_adjusted_fsnow": (
        "Snow fraction adjusted for shade, canopy, and glacier ice"
    ),
    "albedo_clean_flat": "Clean-snow albedo for flat geometry",
    "albedo_dirty_flat": "Dust-affected snow albedo for flat geometry",
    "albedo_clean_terrain_corrected": (
        "Clean-snow albedo corrected for local terrain illumination"
    ),
    "albedo_dirty_terrain_corrected": (
        "Dust-affected snow albedo corrected for local terrain illumination"
    ),
    "delta_vis": "Visible albedo reduction due to snow impurities",
    "radiative_forcing": "Snow radiative forcing due to impurities",
}

_QA_VARIABLES = frozenset(c.INVERSION_EXCLUSION_VARIABLES)
_RESULTS_SUBSET_SCENE_VARIABLES = _QA_VARIABLES | {"spatial_ref"}
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ProductIdentity:
    """Scientific identity required for every persisted product."""

    sensor: str
    product: str
    spatial_id: str
    acquisition_time: str
    platform: Optional[str] = None


@dataclass(frozen=True)
class PersistedProductMetadata:
    """Scientific completion metadata carried by a persisted product."""

    identity: ProductIdentity
    content_profile: str
    product_contents: str = PRODUCT_CONTENTS_FULL
    completed_stages: Tuple[str, ...] = ("invert",)
    completed_operations: Tuple[str, ...] = ()
    completion_status: str = COMPLETION_STATUS_COMPLETE
    grid_digest: str = ""
    created_at: str = ""
    updated_at: str = ""
    package_versions: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = PERSISTED_SCHEMA_VERSION


def validate_persisted_metadata(metadata: PersistedProductMetadata) -> None:
    """Validate versioned file identity and completion metadata."""
    violations = []
    if not isinstance(metadata, PersistedProductMetadata):
        raise_if_violations(
            "persisted_metadata",
            [
                "metadata must be a PersistedProductMetadata instance, "
                f"got {type(metadata).__name__}"
            ],
        )

    if metadata.schema_version != PERSISTED_SCHEMA_VERSION:
        violations.append(
            "schema_version must be "
            f"{PERSISTED_SCHEMA_VERSION}, got {metadata.schema_version!r}"
        )
    if metadata.content_profile not in CONTENT_PROFILES:
        violations.append(
            f"content_profile must be one of {CONTENT_PROFILES!r}, "
            f"got {metadata.content_profile!r}"
        )
    if metadata.product_contents not in PRODUCT_CONTENTS:
        violations.append(
            f"product_contents must be one of {PRODUCT_CONTENTS!r}, "
            f"got {metadata.product_contents!r}"
        )
    if metadata.completion_status != COMPLETION_STATUS_COMPLETE:
        violations.append(
            f"completion_status must be {COMPLETION_STATUS_COMPLETE!r}"
        )

    identity = metadata.identity
    if not isinstance(identity, ProductIdentity):
        violations.append(
            f"identity must be ProductIdentity, got {type(identity).__name__}"
        )
    else:
        for name in ("sensor", "product", "spatial_id", "acquisition_time"):
            _check_nonempty_string(
                getattr(identity, name),
                f"identity.{name}",
                violations,
            )
        if identity.platform is not None:
            _check_nonempty_string(
                identity.platform,
                "identity.platform",
                violations,
            )
        _check_iso_timestamp(
            identity.acquisition_time,
            "identity.acquisition_time",
            violations,
            require_timezone=False,
        )

    expected_stages = {
        CONTENT_PROFILE_INVERSION_RAW: ("invert",),
        CONTENT_PROFILE_POSTPROCESSED_RAW: ("invert", "albedo"),
    }.get(metadata.content_profile)
    if expected_stages is not None and metadata.completed_stages != expected_stages:
        violations.append(
            f"completed_stages must be {expected_stages!r} for "
            f"{metadata.content_profile!r}"
        )

    operations = metadata.completed_operations
    if len(set(operations)) != len(operations):
        violations.append("completed_operations contains duplicate values")
    unknown_operations = tuple(
        operation
        for operation in operations
        if operation not in COMPLETED_OPERATION_ORDER
    )
    if unknown_operations:
        violations.append(
            f"completed_operations contains unknown values {unknown_operations!r}"
        )
    expected_operation_order = tuple(
        operation for operation in COMPLETED_OPERATION_ORDER if operation in operations
    )
    if operations != expected_operation_order:
        violations.append(
            "completed_operations must follow canonical order "
            f"{COMPLETED_OPERATION_ORDER!r}"
        )
    if (
        metadata.content_profile == CONTENT_PROFILE_INVERSION_RAW
        and operations
    ):
        violations.append(
            "inversion_raw products must not list completed postprocessing operations"
        )
    if (
        metadata.content_profile == CONTENT_PROFILE_POSTPROCESSED_RAW
        and not operations
    ):
        violations.append(
            "postprocessed_raw products must list at least one completed operation"
        )

    if not isinstance(metadata.grid_digest, str) or not _HEX_SHA256.fullmatch(
        metadata.grid_digest
    ):
        violations.append("grid_digest must be a lowercase SHA-256 hexadecimal value")
    _check_iso_timestamp(
        metadata.created_at,
        "created_at",
        violations,
        require_timezone=True,
    )
    _check_iso_timestamp(
        metadata.updated_at,
        "updated_at",
        violations,
        require_timezone=True,
    )

    if not isinstance(metadata.package_versions, Mapping):
        violations.append("package_versions must be a mapping")
    else:
        for name, value in metadata.package_versions.items():
            if not isinstance(name, str) or not name.strip():
                violations.append(
                    "package_versions keys must be nonempty strings"
                )
            if not isinstance(value, str) or not value.strip():
                violations.append(
                    f"package_versions[{name!r}] must be a nonempty string"
                )
        for required_package in ("spires-contract", "spires-io"):
            if required_package not in metadata.package_versions:
                violations.append(
                    f"package_versions must include nonempty {required_package!r}"
                )
    if not isinstance(metadata.provenance, Mapping):
        violations.append("provenance must be a mapping")

    raise_if_violations("persisted_metadata", violations)


def validate_persisted_data(
    data: SpiresData,
    metadata: PersistedProductMetadata,
) -> None:
    """Validate one in-memory object against its persisted content profile."""
    validate_persisted_metadata(metadata)
    validate_spires_data(data)
    validate_persisted_grid(data.scene)

    violations = []
    violations.extend(_persisted_group_grid_violations(data))
    missing_qa = tuple(
        name for name in c.INVERSION_EXCLUSION_VARIABLES if name not in data.scene
    )
    if missing_qa:
        violations.append(
            "persisted products require the complete packed QA set; "
            f"missing {missing_qa!r}"
        )

    if data.results is None:
        violations.append("persisted completed products require data.results")
    else:
        try:
            validate_results(data.results, scene=data.scene)
        except ContractError as exc:
            violations.append(str(exc))

    if metadata.product_contents == PRODUCT_CONTENTS_FULL:
        try:
            validate_for_inversion(data)
        except ContractError as exc:
            violations.append(str(exc))
    elif metadata.product_contents == PRODUCT_CONTENTS_RESULTS_SUBSET:
        if data.background is not None:
            violations.append("results_subset products must omit data.background")
        if data.ancillary is not None:
            violations.append("results_subset products must omit data.ancillary")
        unexpected_scene_variables = tuple(
            name
            for name in data.scene.data_vars
            if name not in _RESULTS_SUBSET_SCENE_VARIABLES
        )
        if unexpected_scene_variables:
            violations.append(
                "results_subset scene contains nonessential data variables "
                f"{unexpected_scene_variables!r}"
            )

    if data.results is not None:
        violations.extend(
            _derived_result_violations(
                data.results,
                metadata.completed_operations,
            )
        )

    raise_if_violations("persisted_data", violations)


def _persisted_group_grid_violations(data):
    reference = data.scene["spatial_ref"]
    reference_transform = reference.attrs.get("GeoTransform")
    if reference_transform is None:
        return []
    reference_parts = _geotransform_parts(reference_transform)
    if reference_parts is None:
        return []

    violations = []
    groups = (
        ("background", data.background),
        ("ancillary", data.ancillary),
        ("results", data.results),
    )
    for name, value in groups:
        if value is None:
            continue
        if "spatial_ref" not in value.coords:
            violations.append(f"{name} is missing scalar 'spatial_ref' grid mapping")
            continue
        spatial_ref = value.coords["spatial_ref"]
        if spatial_ref.ndim != 0:
            violations.append(f"{name} 'spatial_ref' must be scalar")
            continue
        transform = spatial_ref.attrs.get("GeoTransform")
        parts = _geotransform_parts(transform)
        if parts is None:
            violations.append(f"{name} 'spatial_ref' has an invalid GeoTransform")
            continue
        tolerance = max(1e-6, max(abs(parts[1]), abs(parts[5])) * 1e-8)
        if not np.allclose(parts, reference_parts, rtol=0.0, atol=tolerance):
            violations.append(
                f"{name} GeoTransform does not match the scene GeoTransform"
            )
    return violations


def validate_persisted_grid(scene) -> None:
    """Validate the self-describing spatial grid required on disk."""
    violations = []
    coordinates = {}
    for name in c.SPATIAL_DIMS:
        if name not in scene.coords:
            violations.append(f"scene is missing coordinate {name!r}")
            continue
        coordinate = scene.coords[name]
        if tuple(coordinate.dims) != (name,):
            violations.append(
                f"coordinate {name!r} must have dimensions {(name,)!r}"
            )
            continue
        values = np.asarray(coordinate.values)
        if values.size == 0:
            violations.append(f"coordinate {name!r} must not be empty")
            continue
        if not np.issubdtype(values.dtype, np.number):
            violations.append(f"coordinate {name!r} must be numeric")
            continue
        if not np.all(np.isfinite(values)):
            violations.append(f"coordinate {name!r} contains non-finite values")
        if values.size > 1:
            differences = np.diff(values)
            if not (np.all(differences > 0) or np.all(differences < 0)):
                violations.append(
                    f"coordinate {name!r} must be strictly monotonic"
                )
        coordinates[name] = values

    if "spatial_ref" not in scene.variables:
        violations.append("scene is missing scalar 'spatial_ref' grid mapping")
    else:
        spatial_ref = scene["spatial_ref"]
        if spatial_ref.ndim != 0:
            violations.append("'spatial_ref' must be scalar")
        crs = (
            spatial_ref.attrs.get("crs_wkt")
            or spatial_ref.attrs.get("spatial_ref")
            or scene.attrs.get("crs_wkt")
        )
        if not isinstance(crs, str) or not crs.strip():
            violations.append(
                "'spatial_ref' or scene metadata must contain a nonempty CRS WKT"
            )
        geotransform = spatial_ref.attrs.get("GeoTransform")
        if geotransform is None:
            violations.append(
                "'spatial_ref' must contain a GeoTransform defining pixel bounds"
            )
        elif all(name in coordinates for name in c.SPATIAL_DIMS):
            violations.extend(
                _geotransform_coordinate_violations(
                    geotransform,
                    x=coordinates["x"],
                    y=coordinates["y"],
                )
            )

    raise_if_violations("persisted_grid", violations)


def _geotransform_coordinate_violations(geotransform, *, x, y):
    parts = _geotransform_parts(geotransform)
    if parts is None:
        return ["'spatial_ref' GeoTransform must contain six numeric values"]
    (
        origin_x,
        pixel_width,
        row_rotation,
        origin_y,
        column_rotation,
        pixel_height,
    ) = parts

    transform = np.asarray(parts, dtype=np.float64)
    if not np.all(np.isfinite(transform)):
        return ["'spatial_ref' GeoTransform contains non-finite values"]
    if pixel_width == 0.0 or pixel_height == 0.0:
        return ["'spatial_ref' GeoTransform pixel sizes must be nonzero"]

    violations = []
    resolution = max(abs(pixel_width), abs(pixel_height))
    tolerance = max(1e-6, resolution * 1e-8)
    if abs(row_rotation) > tolerance or abs(column_rotation) > tolerance:
        violations.append(
            "'spatial_ref' GeoTransform must describe a north-up rectilinear grid"
        )

    expected_x = origin_x + (np.arange(x.size, dtype=np.float64) + 0.5) * pixel_width
    expected_y = origin_y + (np.arange(y.size, dtype=np.float64) + 0.5) * pixel_height
    if not np.allclose(x, expected_x, rtol=0.0, atol=tolerance):
        violations.append(
            "coordinate 'x' must contain the pixel centers implied by "
            "'spatial_ref' GeoTransform"
        )
    if not np.allclose(y, expected_y, rtol=0.0, atol=tolerance):
        violations.append(
            "coordinate 'y' must contain the pixel centers implied by "
            "'spatial_ref' GeoTransform"
        )
    return violations


def _geotransform_parts(geotransform):
    if isinstance(geotransform, str):
        parts = geotransform.split()
    else:
        try:
            parts = list(geotransform)
        except TypeError:
            return None

    if len(parts) != 6:
        return None
    try:
        return tuple(float(value) for value in parts)
    except (TypeError, ValueError):
        return None


def _derived_result_violations(results, operations):
    violations = []
    reference = results.get(c.RESULT_VARIABLES[0])
    for operation in operations:
        for name in OPERATION_RESULT_VARIABLES[operation]:
            if name not in results:
                violations.append(
                    f"completed operation {operation!r} requires result {name!r}"
                )
                continue
            array = results[name]
            violations += [
                f"{name}: {message}"
                for message in (
                    check_dims_present(array, c.RESULT_DIMS)
                    + check_no_extra_dims(array, c.RESULT_DIMS)
                    + check_dims_order(array, c.RESULT_DIMS)
                    + check_dtype(array, c.ACCEPTED_DTYPES)
                    + check_coords_present(array, c.SPATIAL_DIMS)
                )
            ]
            if reference is not None:
                violations += [
                    f"{name}: {message}"
                    for message in check_coords_match(
                        reference,
                        array,
                        c.SPATIAL_DIMS,
                        reference_name=c.RESULT_VARIABLES[0],
                        candidate_name=name,
                    )
                ]
            expected_units = DERIVED_RESULT_UNITS[name]
            if array.attrs.get("units") != expected_units:
                violations.append(
                    f"{name} attribute 'units' must be {expected_units!r}"
                )
            expected_long_name = DERIVED_RESULT_LONG_NAMES[name]
            if array.attrs.get("long_name") != expected_long_name:
                violations.append(
                    f"{name} attribute 'long_name' must be {expected_long_name!r}"
                )

            values = np.asarray(array.values)
            if np.issubdtype(values.dtype, np.floating):
                if np.any(np.isinf(values)):
                    violations.append(f"{name} contains infinite value(s)")
                finite = values[np.isfinite(values)]
                if name != "radiative_forcing" and np.any(finite < 0):
                    violations.append(f"finite {name} values must be nonnegative")
                if (
                    name
                    in {
                        "canopy_adjusted_fsnow",
                        "ice_adjusted_fsnow",
                        *ALBEDO_RESULT_VARIABLES,
                    }
                    and np.any(finite > 1)
                ):
                    violations.append(f"finite {name} values must lie within [0, 1]")
    return violations


def _check_nonempty_string(value, name, violations):
    if not isinstance(value, str) or not value.strip():
        violations.append(f"{name} must be a nonempty string")


def _check_iso_timestamp(value, name, violations, *, require_timezone):
    if not isinstance(value, str) or not value.strip():
        violations.append(f"{name} must be a nonempty ISO-8601 string")
        return
    normalized = value.strip()
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        violations.append(f"{name} must be ISO-8601, got {value!r}")
        return
    if require_timezone and parsed.tzinfo is None:
        violations.append(f"{name} must include a timezone")
