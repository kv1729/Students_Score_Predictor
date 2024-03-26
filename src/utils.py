import sys
import os
import dill
from sklearn.metrics import r2_score

from src.exceptions import CustomException

def save_object(file_path, obj):
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save object to file
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # Raise CustomException with caught exception and sys module
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)  # Train the model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
    except Exception as e:
        # Raise CustomException with caught exception and sys module
        raise CustomException(e, sys)
