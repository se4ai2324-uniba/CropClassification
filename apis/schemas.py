
'''module to define the request and response body for the API'''

# Import the required libraries
from pydantic import BaseModel


class CropFeatures(BaseModel):
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    Temperature: float
    Humidity: float
    pH_Value: float
    Rainfall: float

# Define the response body
class DataOut(BaseModel):
    prediction: str

# Define the response body for model performance
class PerformanceMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
# Definet the Prediction Response
class PredictionResponse(BaseModel):
    predicted: str
    message: str = "Prediction successful"