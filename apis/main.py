
# apis with endpoints predict to predict models
from fastapi import FastAPI,HTTPException, status
import json
from typing import Dict
import joblib
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from prometheus_client import Counter, Summary
from .schemas import CropFeatures, PerformanceMetrics, PredictionResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from prometheus_fastapi_instrumentator import Instrumentator

import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Prometheus Instrumentator
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_count', 'Total number of requests received')

data = pd.read_csv('data/raw/Crop_Recommendation.csv')

try:
    model = joblib.load('models/model.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file not found. Please train the model first.")
    raise
# Root endpoint
@app.get("/", tags=["Root"])
@REQUEST_TIME.time()
def read_root():

    """
    Root endpoint providing basic information about the API.
    """
    return {"message": "Welcome to the Crop Classification API!",
            "name": "Crop Classification API",
            "version": "0.1.2",
            "github": " https://github.com/se4ai2324-uniba/CropClassification.git"}

# Data summary endpoint
@app.get("/summary/",status_code=status.HTTP_200_OK, description="Get summary statistics of the dataset.")
@REQUEST_TIME.time()
def get_summary():
    """
    Get summary statistics of the dataset.

    """
    return data.describe().to_dict()


# Load the model and label encoder
#model, label_encoder = load_model()


# Define performance endpoint
@app.get("/performance",
         status_code= status.HTTP_200_OK,response_model=PerformanceMetrics)
async def get_performance():
    # Example performance metrics
    # Replace these with the actual performance metrics
    with open('models/metrics.json') as f:
        performance_data = json.load(f)

    return PerformanceMetrics(**performance_data)


# Define the predict endpoint
@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(crop_input: CropFeatures):
    try:
            input_data = pd.DataFrame([crop_input.dict()])

            prediction = model.predict(input_data)
            predicted_crop = label_encoder.inverse_transform(prediction)[0]


            return PredictionResponse(
                predicted=predicted_crop,
                input_data=crop_input,

            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Define model info endpoint
@app.get("/model-info")
async def model_info():
    if isinstance(model, RandomForestClassifier):
        return {
            "model_name": "Random Forest Classifier",
            "n_estimators": model.n_estimators,
            "random_state": model.random_state,
            "min_samples_split": model.min_samples_split,
            "max_depth": model.max_depth
        }
    REQUEST_COUNT.inc()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)