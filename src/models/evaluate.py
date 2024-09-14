
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import joblib
from sklearn.metrics import classification_report
import json
import pandas as pd

'''making evalution function'''
def evaluate():


    # Load the model
    model = joblib.load('models/model.pkl')
    
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
    precision = precision_score(y_test, y_pred,average='macro',zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')


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
    
    #report = classification_report(y_test, y_pred,zero_division=0)

    # Save metrics to a JSON file


    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    return metrics
if __name__ == '__main__':
    evaluate()