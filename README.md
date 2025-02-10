# Diabetes Prediction Model

## Project Overview
This project aims to predict whether an individual has diabetes based on a set of medical diagnostic measures. We use the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data) to build a machine learning model that predicts the likelihood of diabetes. The project involves multiple stages, including data preprocessing, model training, evaluation, and creating an interface using Streamlit to allow users to input values and receive predictions.

## Dataset
The dataset used is the Pima Indians Diabetes Database, available on Kaggle. It contains medical data about Pima Indian women and includes the following features:

- Pregnancies: Number of pregnancies.
- Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
- BloodPressure: Diastolic blood pressure (mm Hg).
- SkinThickness: Skinfold thickness (mm).
- Insulin: 2-Hour serum insulin (mu U/ml).
- BMI: Body mass index (weight in kg / (height in m)^2).
- DiabetesPedigreeFunction: A function that represents the likelihood of diabetes based on family history.
- Age: Age of the individual (years).
- Outcome: Whether the individual has diabetes (1) or not (0).

You can download the dataset [here](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data).

## Model Building

### 1. Data Loading and Preprocessing
The data is loaded into a pandas dataframe, and some preprocessing steps are carried out, such as:

- Dropping the first row of the dataset (if necessary).
- Handling zero values: Replacing zero values in specific columns (Glucose, BloodPressure, SkinThickness, Insulin, and BMI) with the median of each column.
- Type Conversion: Columns that should be float are explicitly converted (e.g., BMI and DiabetesPedigreeFunction), and other columns are converted to integers.
- Feature Scaling: Features are scaled using StandardScaler to ensure that the model performs well with standardized data.

### 2. Model Training
We use Random Forest Classifier, Decision Tree and Support Vector Classifier (SVC) for training the model and compare its performance. The dataset is split into training and testing sets using an 80/20 split.

### 3. Model Evaluation
The models are evaluated using various metrics, including:

- Accuracy
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix (Visualized using seaborn)

### 4. Saving the Model
The trained models and the scaler are saved to disk using pickle for later use in a web interface.

## Web Interface (Streamlit)
To allow users to interact with the model, a Streamlit app is created. The app takes inputs from users for various health metrics (such as glucose, age, BMI) and provides a prediction based on the trained model.

### Streamlit Setup
The Streamlit app loads the saved model and scaler, takes inputs from the user, and then provides a prediction with the probability of diabetes.

### How to Run the Application
 
#### Clone the repository to your local machine:
```bash
git clone https://github.com/WillianIsami/diabetes_prediction.git
cd diabetes_prediction
```

#### Install the required dependencies by running:
```bash
pip install -r requirements.txt
```
or create a new environment with conda
```bash
conda env create -f environment.yml
conda activate diabetes_prediction
```

#### Run the Streamlit application:
```bash
streamlit run main.py
```
The interface will open in your browser, where you can input values and predict the likelihood of diabetes.