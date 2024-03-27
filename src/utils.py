import os
import sys
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save object to file using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Raise CustomException with caught exception and sys module
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        # Iterate over models and their corresponding parameters
        for model_name, model in models.items():
            # Extract parameters for the current model
            model_params = params.get(model_name, {})

            # Perform grid search for hyperparameter tuning
            gs = GridSearchCV(model, model_params, cv=3)
            gs.fit(X_train, y_train)

            # Set best parameters found by GridSearchCV to the model
            model.set_params(**gs.best_params_)

            # Train the model with tuned hyperparameters
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R^2 score for training and test data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test R^2 score in the report dictionary
            report[model_name] = test_model_score

        return report

    except Exception as e:
        # Raise CustomException with caught exception and sys module
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        # Load object from file using pickle
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        # Raise CustomException with caught exception and sys module
        raise CustomException(e, sys)
