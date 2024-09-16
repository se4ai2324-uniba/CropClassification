
"""
This module contains functions for evaluating machine learning models.

It includes functions for calculating performance metrics such as accuracy,
precision, recall, and F1-score, as well as visualizing model predictions.
"""

import json
from sklearn.metrics import (accuracy_score,
                             precision_score, recall_score, f1_score)
import joblib
import mlflow
import dagshub
import pandas as pd


def evaluate():

    """
    Evaluate the model's predictions against the true values.

    Returns:
        float: Accuracy score of the model's predictions.
    """
    dagshub.init(repo_owner='ushafique',
                 repo_name='CropClassification',
                 mlflow=True)

    mlflow.set_experiment('Crop Classification Evaluations')
    # Load the model
    mlflow.start_run()
    model = joblib.load('models/model.pkl')

    mlflow.sklearn.log_model(model, "model")

    # Load the test data
    test_data = pd.read_csv('data/processed/test.csv')

    # Split the data into features (X) and target (y)
    X_test = test_data.drop('Crop', axis=1)
    y_test = test_data['Crop']

    label_encoder = joblib.load('models/label_encoder.pkl')

    # Make predictions
    y_pred_encoded = model.predict(X_test)

    # Convert predictions back to original labels
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    # Calculate evaluation metrics

    # Make predictions
    # y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test,
                                y_pred,
                                average='macro',
                                zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Print metrics
    print(f"accuracy: {accuracy}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Save the evaluation metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    mlflow.end_run()

    # Save metrics to a JSON file
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f)

    return metrics


if __name__ == '__main__':
    evaluate()
