from pydantic import BaseModel
from typing import Dict, Any


class PredictRequest(BaseModel):
    """
    Input data received by the API.
    Avazu features are provided as key-value pairs.
    """
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    """
    Output returned by the API.
    """
    click_probability: float
    click_prediction: int
