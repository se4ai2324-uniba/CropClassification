
# apis with endpoints predict to predict models
# """
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data = pd.read_csv('data/raw/Crop_Recommendation.csv')


# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Classification API!"}


@app.get("/summary/")
def get_summary():
    """Get summary statistics of the dataset."""
    return data.describe().to_dict()


# Load the model and label encoder
model = joblib.load('models/model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

try:

    model = joblib.load('models/model.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
except FileNotFoundError:
    logger.error("Model file not found. Please train the model first.")

# Define the request body

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

# Define the predict endpoint
@app.post("/predict")
def predict(crops: CropFeatures):
    try:

    # Convert the input data to a pandas DataFrame
        input_data = pd.DataFrame([[crops.Nitrogen,
                                crops.Phosphorus,
                                crops.Potassium,
                                crops.Temperature,
                                crops.Humidity,
                                crops.pH_Value,
                                    crops.Rainfall]])

        # Make predictions
        prediction_encoded = model.predict(input_data)
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]

        # Return the prediction
        return {"prediction": prediction}
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

