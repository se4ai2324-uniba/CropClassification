from locust import HttpUser, task, between
import random
import pandas as pd

# Load the crop recommendation dataset
df = pd.read_csv('data/raw/Crop_Recommendation.csv')  # Adjust the path as necessary

class CropRecommendationUser(HttpUser):
    wait_time = between(1, 3)  # Wait time between tasks (in seconds)

    @task(1)
    def get_info(self):
        self.client.get('/', name = 'Get_info')

    @task(2)
    def get_model_info(self):
        self.client.get('/model-info', name = 'Get_Model_Info')


    @task(1)
    def get_prediction(self):
        payload = {
            "Nitrogen": 29,
            "Phosphorus": 89,
            "Potassium": 32,
            "Temperature": 42.89,
            "Humidity": 67.0,
            "pH_Value": 6.5,
            "Rainfall": 201.2
        }
        self.client.post("/predict", json=payload, name = 'Get_Prediction')

    # @task
    # def get_crop_recommendation(self):
    #     # Randomly select a row from the DataFrame to simulate user input
    #     row = df.sample().iloc[0]

    #     # Prepare the query parameters for the GET request
    #     params = {
    #         "Nitrogen": row['Nitrogen'],
    #         "Phosphorus": row['Phosphorus'],
    #         "Potassium": row['Potassium'],
    #         "Temperature": row['Temperature'],
    #         "Humidity": row['Humidity'],
    #         "pH_Value": row['pH_Value'],
    #         "Rainfall": row['Rainfall']
    #     }

    #     # Make a GET request to the crop recommendation endpoint
    #     self.client.post("/predict", params=params)