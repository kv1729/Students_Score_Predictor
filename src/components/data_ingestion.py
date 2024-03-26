import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exceptions import CustomException  # Importing CustomException from exceptions module
import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

@dataclass
class DataPaths:
    train_path: str
    test_path: str

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv('notebook/data/stud.csv')  # Assuming data path is corrected
            logging.info(f'Read the dataset as df (shape: {df.shape})')  # Using f-string

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of the data completed')

            return DataPaths(train_path=self.ingestion_config.train_data_path, 
                             test_path=self.ingestion_config.test_data_path)

        except Exception as e:
            logging.error(f"An error occurred during data ingestion: {e}")
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    # Assuming logging configuration is done elsewhere
    obj = DataIngestion()
    obj.initiate_data_ingestion()
