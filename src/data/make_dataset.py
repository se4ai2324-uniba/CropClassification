# -*- coding: utf-8 -*-
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import pandas as pd
import os
from sklearn.model_selection import train_test_split



# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     #load_dotenv(find_dotenv())

#     main()
 
def process_data():
    # Define the path to the raw data
    raw_data_path = os.path.join('data', 'raw', 'Crop_Recommendation.csv')
    
    # Check if the raw data file exists
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"The file {raw_data_path} does not exist. Please check the path.")

    # Read the raw data
    df = pd.read_csv(raw_data_path)
    
    # Split the data into features (X) and target (y)
    X = df.drop('Crop', axis=1)  # Assuming 'Crop' is the target variable
    y = df['Crop']
    
    # Split the data into train and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Combine features and target for train and test sets
    train_data = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    
    # Create processed data directory if it doesn't exist
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save train and test data to CSV files
    train_data.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)
    
    print("Data processed and saved to data/processed/train.csv and data/processed/test.csv")

if __name__ == "__main__":
    process_data()
