import numpy as np
from alibi_detect.cd import TabularDrift
from sklearn.model_selection import train_test_split
import pandas as pd


# Load the data
data = pd.read_csv('data/raw/Crop_Recommendation.csv')

# Split the data into features (X) and target (y)
X = data.drop('Crop', axis=1)
y = data['Crop']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Initialize the TabularDrift detector
drift_detector = TabularDrift(X_train.to_numpy(), p_val=0.05)  # p_val is the threshold for drift detection

# Check for data drift
drift_result = drift_detector.predict(X_test.to_numpy(), drift_type="batch")

if drift_result['data']['is_drift']:
    print("Data drift detected!")
else:
    print("No data drift detected.")
