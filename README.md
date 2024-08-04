# Diabetes Prediction

This project involves predicting diabetes using the Pima Indians Diabetes dataset. It includes data preprocessing and the training and evaluation of several machine learning models.

## Loading the Dataset

1. **Download the Dataset**: The Pima Indians Diabetes dataset was downloaded from Kaggle and unzipped for analysis.

2. **Data Loading**: The dataset was loaded into a Pandas DataFrame for initial inspection.

3. **Data Inspection**: Basic statistics and data types were examined. Zero values were checked, and the data was sorted by the outcome variable.

## Data Preprocessing

1. **Handling Zero Values**: 
    - The dataset was divided into two groups based on the outcome variable (0 and 1).
    - For each group, zero values in the features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI) were replaced with median or mean values, depending on the feature and outcome.

2. **Data Reassembly**: The processed groups were combined back into a single DataFrame.

## Model Training

1. **Train-Test Split**: The data was split into training and testing sets to evaluate the models' performance.

2. **Model Training and Evaluation**:
    - **Logistic Regression**: Trained and evaluated to assess its accuracy.
        - Accuracy 78.35497835497836
    - **K-Nearest Neighbors (KNN)**: Trained with varying `k` values to find the optimal `k` and evaluated its accuracy.

        - Default k: Accuracy 82.68398268398268
        - K = 35: Accuracy 98
    - **Decision Tree**: Trained and evaluated. 
        - Accuracy 86.58008658008657
    - **Naive Bayes**: Trained and evaluated.
        - Accuracy 75.32467532467533
    - **Support Vector Machine (SVM)**: Trained and evaluated.
        - Accuracy: 80.08658008658008
    - **Random Forest**: Trained and evaluated.
        - Accuracy: 87.44588744588745
    - **Gradient Boosting**: Trained and evaluated.
        - Accuracy: 90.9090909090909
    - **AdaBoost**: Trained and evaluated.
        - Accuracy: 87.44588744588745
    - **Extra Trees**: Trained and evaluated.
        - Accuracy: 85.28138528138528
    - **CatBoost**: Trained and evaluated.
        - Accuracy: 85.71428571428571
    - **XGBoost**: Trained and evaluated, including hyperparameter tuning using GridSearchCV.
        - Accuracy: 72.72727272727273

3. **Model Performance**: Performance metrics such as accuracy were computed for each model. Additionally, for KNN and Random Forest, detailed analysis including classification reports and confusion matrices were generated to evaluate model performance.

## Results

- Various machine learning models were compared based on their accuracy scores.
- The results were visualized using error rate plots and confusion matrices to understand the performance of each model.
