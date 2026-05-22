"""
schemas.py
==========
Pydantic models for the MediShield AI prediction API.

These models define and validate the shape of every request and response
that passes through the POST /predict endpoint.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Request schema  — sent by the frontend prediction form
# ---------------------------------------------------------------------------

class PatientData(BaseModel):
    """
    Eight clinical features required by the DiabetesRiskPredictor model.
    Field names use snake_case to match the POST body the frontend sends.
    """

    pregnancies: float = Field(
        ..., ge=0, le=20,
        description="Number of times pregnant"
    )
    glucose: float = Field(
        ..., ge=0, le=300,
        description="Plasma glucose concentration (mg/dL)"
    )
    blood_pressure: float = Field(
        ..., ge=0, le=200,
        description="Diastolic blood pressure (mmHg)"
    )
    skin_thickness: float = Field(
        ..., ge=0, le=100,
        description="Triceps skin fold thickness (mm)"
    )
    insulin: float = Field(
        ..., ge=0, le=900,
        description="2-hour serum insulin (μU/mL)"
    )
    bmi: float = Field(
        ..., ge=0, le=70,
        description="Body mass index (kg/m²)"
    )
    diabetes_pedigree_function: float = Field(
        ..., ge=0, le=3,
        description="Diabetes pedigree function score"
    )
    age: float = Field(
        ..., ge=0, le=120,
        description="Age in years"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "pregnancies": 2,
                "glucose": 180,
                "blood_pressure": 85,
                "skin_thickness": 30,
                "insulin": 150,
                "bmi": 34.5,
                "diabetes_pedigree_function": 0.72,
                "age": 52,
            }
        }


# ---------------------------------------------------------------------------
# Response schema  — returned to the frontend
# ---------------------------------------------------------------------------

class ShapValueItem(BaseModel):
    feature: str
    value: float

class PredictionResult(BaseModel):
    """
    Prediction output returned by POST /predict.
    """

    risk_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Predicted diabetes probability (0.0 = no risk, 1.0 = certain)"
    )
    risk_level: str = Field(
        ...,
        description="Risk category: LOW (< 40 %), MEDIUM (40–70 %), HIGH (> 70 %)"
    )
    top_factors: List[str] = Field(
        ...,
        description="Top 3 clinical features driving this prediction (gradient-based SHAP)"
    )
    
    # New fields for fully dynamic SHAP dashboard support
    prediction: Optional[str] = Field(
        default=None,
        description="Risk level string (e.g. 'High Risk' or 'Low Risk')"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence percentage (0.0 to 100.0)"
    )
    risk_factors: Optional[float] = Field(
        default=None,
        description="Sum of positive SHAP feature importances as percentage"
    )
    protective_factors: Optional[float] = Field(
        default=None,
        description="Sum of negative SHAP feature importances as percentage"
    )
    shap_values: Optional[List[ShapValueItem]] = Field(
        default=None,
        description="SHAP feature importance array for Recharts"
    )
    shap_values_dict: Optional[dict] = Field(
        default=None,
        description="Dictionary mapping features to SHAP values"
    )
    probability_percent: Optional[float] = Field(
        default=None,
        description="Probability as percentage (0.0 to 100.0)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "risk_score": 0.91,
                "risk_level": "HIGH",
                "top_factors": ["Glucose Level", "BMI", "Age"],
                "prediction": "High Risk",
                "confidence": 91.0,
                "risk_factors": 113.0,
                "protective_factors": -13.0,
                "shap_values": [
                    {"feature": "Glucose Level", "value": 35.0},
                    {"feature": "BMI", "value": 28.0},
                    {"feature": "Blood Pressure", "value": 18.0},
                    {"feature": "Insulin", "value": 14.0},
                    {"feature": "Age", "value": 12.0},
                    {"feature": "Heart Rate", "value": -8.0}
                ]
            }
        }

