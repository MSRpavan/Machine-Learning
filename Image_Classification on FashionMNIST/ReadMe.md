# Hand Written Digits Recognition using ANN on MNIST

This project aims to predict the likelihood of heart disease in patients using various classification machine learning models. The models used include K-Nearest Neighbors (KNN), Logistic Regression, and Random Forest Classifier. The dataset used in this project contains information about patients' age, sex, chest pain type, heart rate, cholesterol levels, and so on , the presence or absence of heart disease (target variable).

## Dataset

The dataset used in this project contains the following columns:

- `age`: Age of the patient
- `sex`: Gender of the patient (0: female, 1: male)
- `chest pain`: Type of chest pain experienced by the patient (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)
- `heart rate`: Heart rate of the patient
- `cholesterol`: Cholesterol levels of the patient and some other as slope , thalach..so on.
- `target`: Presence or absence of heart disease (0: no heart disease, 1: heart disease)

## Data Preprocessing

Before building the machine learning models, the dataset underwent preprocessing steps such as handling missing values, encoding categorical variables, and scaling numerical features. Additionally, hyperparameters were tuned to optimize the performance of the models.

## Models Used

Three classification models were employed in this project:

1. **K-Nearest Neighbors (KNN)**: A non-parametric algorithm that classifies a data point based on the majority class of its k nearest neighbors.
2. **Logistic Regression**: A linear model used for binary classification tasks, which predicts the probability of a binary outcome.
3. **Random Forest Classifier**: An ensemble learning method that constructs a multitude of decision trees during training and outputs the mode of the classes as the prediction.

## Evaluation

The performance of each model was evaluated using metrics such as accuracy, precision, recall, and F1-score. Cross-validation techniques were employed to ensure robustness of the models' performance.

## Requirements

- Python (>=3.6)
- scikit-learn
- pandas
- matplotlib
- numpy

## Contributors

- [M.S.R.Pavan](https://github.com/MSRpavan)

