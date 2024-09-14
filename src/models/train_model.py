import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
import joblib
##function to train the model

def train():
    output_dir = "models"
    output_file = os.path.join(output_dir, "model.pkl")

# Ensure the models directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the processed data
    train_data = pd.read_csv('data/processed/train.csv')
    
    # Split the data into features (X) and target (y)
    X_train = train_data.drop('Crop', axis=1)
    y_train = train_data['Crop']
    
    # Create a label encoder object
    label_encoder = LabelEncoder()
    label_encoded=label_encoder.fit_transform(y_train)

    # Initialize the mode
    model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=5,max_depth=2)
    
    # Train the model
    model.fit(X_train, label_encoded)


    # Save the model
    
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    joblib.dump(model, output_file)


if __name__ == "__main__":
    train()