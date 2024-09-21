"""
This module contains functions for training machine learning models.

It includes functions for loading data training data sets,
and training a Random Forest model using the scikit-learn library.
"""
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import mlflow.sklearn
import dagshub
import joblib
from codecarbon import EmissionsTracker


# function to train the model


def train():

    """
    Train a Random Forest model on the given dataset.

    Returns:
        model: Trained Random Forest model.
    """

    output_dir = "models"
    output_file = os.path.join(output_dir, "model.pkl")

# Ensure the models directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dagshub.init(
        repo_owner='ushafique',
        repo_name='CropClassification',
        mlflow=True)

    # applying mlflow autologging
    mlflow.sklearn.autolog()
    # Load the processed data
    train_data = pd.read_csv('data/processed/train.csv')

    # Split the data into features (X) and target (y)
    X_train = train_data.drop('Crop', axis=1)
    y_train = train_data['Crop']

    # Create a label encoder object
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(y_train)

    # Initialize emissions tracker
    tracker = EmissionsTracker()

    tracker.start()
    # Initialize the mode
    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        min_samples_split=5,
        max_depth=2)

    # Train the model
    model.fit(X_train, label_encoded)

    # Save the model
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    joblib.dump(model, output_file)
    print("Model saved to: models/model.pkl")

    emissions = tracker.stop()


    # Save the emissions report to in text file
    with open('reports/train_model_emissions_report.txt', 'w') as f:
        f.write(f"Estimated Carbon Emission for Model Training: {emissions:.5f} kg CO2\n")
        f.write(f"Estimated Energy Consumption for Model Training: {emissions * 0.000055}\n")


if __name__ == "__main__":
    train()
