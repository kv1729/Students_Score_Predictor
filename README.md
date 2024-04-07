# Description: 
# Students Marks Predictor (End to end machine learning project)



## Description

The Students Marks Predictor is an end-to-end project developed using Python inside a Conda environment in Visual Studio Code. This project aims to predict the marks of students based on various criteria provided in the dataset. It serves both as a tool for predicting student marks and as a learning resource to understand the workflow of developing a machine learning model.

## Workflow

1. **Data Preprocessing**: The project starts with the preprocessing of the dataset, which involves tasks such as handling missing values, encoding categorical variables, and feature scaling.

2. **Model Training**: Several machine learning models are trained using the preprocessed data, including Random Forest, Decision Tree, Gradient Boosting, Linear Regression, XGBRegressor, CatBoosting Regressor, and AdaBoost Regressor.

3. **Model Evaluation**: The performance of each model is evaluated using metrics such as accuracy, and the best-performing model is selected for deployment.

4. **Deployment**: The selected model is deployed using Flask, creating a web application where users can input criteria and receive predicted marks for students.

## Usage

To use the Students Marks Predictor:

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Run the Flask application using `python app.py`.
5. Access the web application in your browser and input the required criteria to predict student marks.

## Contributing

Contributions to the project are welcome! You can contribute by:
- Adding new features or improvements to the codebase.
- Enhancing the model's accuracy or efficiency.
- Providing suggestions for workflow enhancements.

Please fork the repository, make your changes, and submit a pull request.

## License

This project is not licensed.

## Troubleshooting

If you encounter any issues or errors while using the application, please refer to the logger file for detailed information on the problem.

## Contact

For any inquiries or feedback, feel free to reach out to [kunal.dixit14@gmail.com](mailto:kunal.dixit14@gmail.com).

## Models and Accuracy

The following machine learning models were trained in this project :
- Random Forest
- Decision Tree
- Gradient Boosting
- Linear Regression
- XGBRegressor
- CatBoosting Regressor
- AdaBoost Regressor

## Best Model accuracy

The trainer selects the best model from the above and gives the accuracy of the best model.
Best Accuracy: 87.62%

## User Application

The web application for this project is built using Flask.

