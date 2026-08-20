
from typing import Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):

    equipment_id: Optional[str] = None
    train: Optional[str] = None

    commission_year: Optional[float] = None
    asset_age_years: Optional[float] = None
    hours_since_maint: Optional[float] = None
    cumulative_op_hours: Optional[float] = None
    bearing_temperature: Optional[float] = None
    rpm: Optional[float] = None
    vibration: Optional[float] = None
    overall_vibration: Optional[float] = None
    motor_current: Optional[float] = None
    oil_pressure: Optional[float] = None
    oil_particles_ppm: Optional[float] = None
    bearing_index: Optional[float] = None
    discharge_pressure: Optional[float] = None
    feed_gas_pressure: Optional[float] = None
    lng_output_tph: Optional[float] = None
    ambient_temperature: Optional[float] = None
    load_factor: Optional[float] = None
    wear_level: Optional[float] = None
    lubrication_health_index: Optional[float] = None
    production_efficiency: Optional[float] = None
    quality_factor: Optional[float] = None


class PredictionResponse(BaseModel):

    equipment_id: Optional[str] = None
    train: Optional[str] = None

    failure_probability_24h: float = Field(
        ge=0.0,
        le=1.0
    )

    failure_risk: str
    rul_days: float

    classification_model: str
    regression_model: str


class HealthResponse(BaseModel):

    status: str
    classification_model: str
    regression_model: str
    feature_count: int
