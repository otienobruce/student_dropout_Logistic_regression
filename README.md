# Student Dropout Prediction (Logistic Regression)
## Project Overview

This project explores student dropout prediction using Machine Learning.
The goal is to identify factors that may influence whether a student drops out or continues their studies.

The project was built as part of my machine learning learning journey to practice:

- Data preprocessing
- Feature encoding
- Binary classification
- Model evaluation using classification metrics

### Objective

To build a Logistic Regression model that predicts whether a student will:

0 → Continue

1 → Dropout

based on academic, behavioral, and demographic features.

### Dataset

The dataset contains student-related features such as:

  GPA
  
  Study Hours per Day
  
  Assignment Delay Days
  
  Stress Index
  
  Parental Education
  
  Department
  
  Internet Access
  
  Gender
  
  Family Income

Dataset source:
https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset

The dataset is not stored in this repository to keep the project lightweight.
#### Data Preprocessing

The following preprocessing steps were performed:

  Handling missing values (median imputation)
  
  Encoding categorical variables:
  
  Label encoding for binary categories
  
  Ordinal encoding for ordered categories
  
  One-hot encoding for departments
  
  Feature selection experiments
  
  Train/Test split using stratified sampling

#### Model Used

Logistic Regression (Binary Classification)

Configuration:

LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

Why Logistic Regression?

Simple baseline model

Interpretable coefficients

Good for binary classification problems

#### Evaluation Metrics

The model was evaluated using:

  Accuracy
  
  Precision
  
  Recall
  
  F1-score
  
  Classification Report

These metrics helped evaluate how well the model predicts both dropout and non-dropout students.

#### Key Findings

Academic performance indicators (e.g., GPA, study hours) influenced dropout predictions.

Stress-related features also showed predictive power.

Using class_weight='balanced' helped improve performance on the minority class.

Feature selection impacted model performance.

### What I Learned

Through this project I learned:

  How to prepare real-world data for machine learning.
  
  Handling numerical vs categorical variables.
  
  Building and evaluating a Logistic Regression model.
  
  Understanding precision, recall, and F1-score.
  
  Importance of stratified train-test splitting.

### Limitations

Feature scaling was not applied (still learning pipelines).

No hyperparameter tuning yet.

Cross-validation not implemented.

Model is a baseline for learning purposes.

### Future Improvements

Implement preprocessing pipelines.

Apply feature scaling.

Try advanced models (Random Forest, XGBoost).

Perform hyperparameter tuning.

Add ROC-AUC and confusion matrix visualization.

#### Tech Stack

  - Python
  
  - Jupyter Notebook
  
  - Pandas
  
  - NumPy
  
  - Matplotlib
  
  - Scikit-learn

#### Author

Bruce Daniel
Computer Science Student | Learning Data Science & Machine Learning
