import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exceptions import CustomException
import logging
from src.components.data_transformation import DataTransformer
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self, logger):
        self.ingestion_config = DataIngestionConfig()
        self.logger = logger

    def initiate_data_ingestion(self):
        self.logger.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv('notebook/data/stud.csv')  # Assuming correct data path
            self.logger.info(f'Read the dataset as df (shape: {df.shape})')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            self.logger.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            self.logger.info('Ingestion of the data completed')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            self.logger.error(f"An error occurred during data ingestion: {e}")
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    # Assuming logging configuration is done elsewhere
    logger = logging.getLogger(__name__)
    obj = DataIngestion(logger)
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformer()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
