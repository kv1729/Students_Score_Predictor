import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object
from src.exceptions import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformer:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[("imputer", SimpleImputer(strategy='median')),
                       ("scaler", StandardScaler())]
            )

            cat_pipeline = Pipeline(
                steps=[("one_hot_encoder", OneHotEncoder())]
            )

            logging.info(f"Categorical columns : {categorical_columns}")
            logging.info(f"Numerical columns : {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtain preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            train_arr = preprocessing_obj.fit_transform(train_df.drop(columns=[target_column_name]))
            test_arr = preprocessing_obj.transform(test_df.drop(columns=[target_column_name]))

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                np.c_[train_arr, train_df[target_column_name]],
                np.c_[test_arr, test_df[target_column_name]],
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataTransformer()
    train_data, test_data, preprocessor_path = obj.initiate_data_transformation(train_path="path/to/train.csv", test_path="path/to/test.csv")
