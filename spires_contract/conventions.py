"""Canonical naming and dtype conventions shared across all SPIReS contracts.

Single source of truth so every boundary module (spectra, lut, r0, results) speaks
the same dimension-name and dtype vocabulary.
"""

import numpy as np

# Canonical single-scene spatial dimensions.
SPATIAL_DIMS = ("y", "x")

# Spatial + spectral spectra arrays (target / background reflectance).
SPECTRA_DIMS = SPATIAL_DIMS + ("band",)

# Per-pixel solar zenith angle in unpacked physical degrees.
SOLAR_ANGLE_DIMS = SPATIAL_DIMS
SOLAR_ANGLE_UNITS = "degrees"
SOLAR_ANGLE_UNIT_ALIASES = ("degree", "degrees", "deg")
SOLAR_ZENITH_RANGE_DEGREES = (0.0, 180.0)

# Cluster-level arrays and the excluded-pixel sentinel used by cluster labels.
CLUSTER_DIM = "cluster"
CLUSTER_DIMS = (CLUSTER_DIM,)
CLUSTER_LABEL_DIMS = SPATIAL_DIMS
CLUSTER_LABEL_SENTINEL = -1

# Legacy normalized MATLAB reflectance layout. Retained only while runtime
# MATLAB LUT support remains; canonical Dataset contracts are defined below.
LUT_DIMS = ("band", "solar_angle", "lap_concentration", "grain_size")

REFLECTANCE_LUT_VARIABLE = "reflectance"
REFLECTANCE_LUT_DIMS = (
    "band",
    "solar_angle",
    "lap_concentration",
    "sqrt_grain_radius",
)
ALBEDO_LUT_VARIABLE = "albedo"
ALBEDO_LUT_REQUIRED_DIMS = (
    "solar_zenith",
    "illumination_angle",
    "lap_concentration",
    "sqrt_grain_radius",
)
ALBEDO_LUT_OPTIONAL_DIMS = ("skyview", "altitude")
LUT_LAP_TYPE_ATTR = "lap_type"
LUT_AXIS_UNITS = {
    "solar_angle": "degrees",
    "solar_zenith": "degrees",
    "illumination_angle": "degrees",
    "lap_concentration": "ppm",
    "sqrt_grain_radius": "um^0.5",
    "skyview": "1",
    "altitude": "km",
}

# Canonical public inversion result variables, in source-vector order.
RESULT_VARIABLES = ("fsnow", "fshade", "lap_concentration", "grain_radius")
RESULT_DIMS = SPATIAL_DIMS

# Required scene variables at the inversion boundary.
REQUIRED_SCENE_VARIABLES = (
    "reflectance",
    "solar_zenith",
)

# Optional atomic inversion-exclusion provenance set. These variables are all
# present or all absent in a validated SpiresData scene.
VALID_INVERSION_MASK_VARIABLE = "valid_inversion_mask"
INVERSION_EXCLUSION_FLAGS_VARIABLE = "inversion_exclusion_flags"
INVERSION_EXCLUSION_ASSESSED_VARIABLE = "inversion_exclusion_assessed"
INVERSION_EXCLUSION_VARIABLES = (
    INVERSION_EXCLUSION_FLAGS_VARIABLE,
    INVERSION_EXCLUSION_ASSESSED_VARIABLE,
    VALID_INVERSION_MASK_VARIABLE,
)
OPTIONAL_SCENE_VARIABLES = INVERSION_EXCLUSION_VARIABLES

# Stable schema-v1 bit assignments. Published assignments must never be
# reordered; new meanings may use only currently reserved bits.
INVERSION_EXCLUSION_SCHEMA_VERSION = 1
INVERSION_EXCLUSION_DTYPE = np.dtype(np.uint16)
INVERSION_EXCLUSION_REASONS = (
    "invalid_reflectance",
    "invalid_geometry",
    "insufficient_observations",
    "poor_surface_reflectance_quality",
    "cloud",
    "cloud_shadow",
    "water",
    "ice",
    "playa",
    "low_reflectance",
    "user_exclusion",
)
INVERSION_EXCLUSION_BITS = {
    reason: 1 << bit_index
    for bit_index, reason in enumerate(INVERSION_EXCLUSION_REASONS)
}
INVERSION_EXCLUSION_KNOWN_MASK = sum(INVERSION_EXCLUSION_BITS.values())
INVERSION_EXCLUSION_RESERVED_BITS = tuple(range(11, 16))
INVERSION_EXCLUSION_SCHEMA_ATTR = "inversion_exclusion_schema_version"

# Required variables for a complete clustered scene.
CLUSTER_LABEL_VARIABLE = "cluster_label"
CLUSTER_COUNT_VARIABLE = "cluster_count"
CLUSTER_REPRESENTATIVE_VARIABLES = (
    "cluster_representative_reflectance",
    "cluster_representative_background",
    "cluster_representative_solar_zenith",
)
REQUIRED_CLUSTER_VARIABLES = (
    CLUSTER_LABEL_VARIABLE,
    CLUSTER_COUNT_VARIABLE,
) + CLUSTER_REPRESENTATIVE_VARIABLES
OPTIONAL_CLUSTER_VARIABLES = ("cluster_representative_cosine_illumination",)
CLUSTER_MASK_POLICY_ATTR = "valid_inversion_mask_applied"

# Canonical result metadata. Additional attributes are allowed.
RESULT_LONG_NAMES = {
    "fsnow": "Fractional snow endmember",
    "fshade": "Fractional shade endmember",
    "lap_concentration": "Light-Absorbing Particle Concentration in Snow",
    "grain_radius": "Effective Snow Grain Radius",
}
RESULT_UNITS = {
    "fsnow": "1",
    "fshade": "1",
    "lap_concentration": "ppm",
    "grain_radius": "um",
}
GRAIN_RADIUS_UNIT_ALIASES = ("um", "µm", "μm")
SUPPORTED_LAP_TYPES = ("dust",)

# Canonical floating dtype for scientific data at SPIReS package boundaries.
# Internal numerical kernels may promote values for computation, but public
# producers emit float32 and validators never silently cast float64 inputs.
ACCEPTED_DTYPES = (np.float32,)

# Back-compat alias: some callers referenced REQUIRED_DTYPE. Kept pointing at the
# accepted set so a single source of truth drives validation.
REQUIRED_DTYPE = ACCEPTED_DTYPES
