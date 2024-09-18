
# apis with endpoints predict to predict models
from fastapi import FastAPI,HTTPException, status
import json
from typing import Dict
import joblib
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from .schemas import CropFeatures, PerformanceMetrics, PredictionResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

data = pd.read_csv('data/raw/Crop_Recommendation.csv')


# Root endpoint
@app.get("/", tags=["Root"])
def read_root():

    """
    Root endpoint providing basic information about the API.
    """
    return {"message": "Welcome to the Crop Classification API!",
            "name": "Crop Classification API",
            "version": "0.1.0",
            "github": " https://github.com/se4ai2324-uniba/CropClassification.git"}

# Data summary endpoint
@app.get("/summary/",status_code=status.HTTP_200_OK, description="Get summary statistics of the dataset.")
def get_summary():
    """
    Get summary statistics of the dataset.

    """
    return data.describe().to_dict()


# Load the model and label encoder
def load_model():
    """
    Load the trained model and label encoder.
    """
    try:

        model = joblib.load('models/model.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        return model, label_encoder
    except FileNotFoundError:
        logger.error("Model file not found. Please train the model first.")

model, label_encoder = load_model()
if model is None or label_encoder is None:
    raise Exception("Model and label encoder not found. Please train the model first.")


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)