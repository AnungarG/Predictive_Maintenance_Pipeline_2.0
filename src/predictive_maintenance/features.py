"""Authoritative feature and target contracts."""

FEATURE_COLUMNS = [
    "commission_year", "asset_age_years", "hours_since_maint",
    "cumulative_op_hours", "bearing_temperature", "rpm", "vibration",
    "overall_vibration", "motor_current", "oil_pressure",
    "oil_particles_ppm", "bearing_index", "discharge_pressure",
    "feed_gas_pressure", "lng_output_tph", "ambient_temperature",
    "load_factor", "wear_level", "lubrication_health_index",
    "production_efficiency", "quality_factor"
]

CLASSIFICATION_TARGET = "failure_within_24h"
REGRESSION_TARGET = "rul_days"

TARGET_COLUMNS = [
    "failure_within_24h", "failure_within_72h", "failure_within_7d",
    "rul_days", "rul_censored"
]

EXCLUDED_FEATURES = [
    "row_id", "timestamp", "train", "equipment_id", "equipment_name",
    "equipment_type", "area", "manufacturer", "criticality",
    "rul_days", "rul_censored", "failure_within_24h",
    "failure_within_72h", "failure_within_7d", "failure_rate"
]

PHYSICAL_BOUNDS = {
    "load_factor": (0, 1),
    "wear_level": (0, 1),
    "lubrication_health_index": (0, 100),
    "oil_pressure": (0, None),
    "oil_particles_ppm": (0, None),
    "overall_vibration": (0, None),
    "vibration": (0, None),
    "rpm": (0, None),
    "hours_since_maint": (0, None),
    "cumulative_op_hours": (0, None),
    "rul_days": (0, None),
}
