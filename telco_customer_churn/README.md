# Telco Customer Churn Prediction

This project builds a machine learning model to predict whether telecom customers will churn (leave the service). The model is built using Logistic Regression and evaluates how customer demographics, services, and billing information influence churn.

## Objective

The objective of this project is to develop a classification model that predicts whether a customer will churn based on their account and service usage information.

## Dataset

The project uses the **Telco Customer Churn dataset**, which contains customer information such as:

- Tenure
- Monthly Charges
- Contract Type
- Internet Service
- Payment Method
- Customer demographics and service subscriptions

The target variable is:

- **Churn**
  - 0 → Customer stays
  - 1 → Customer leaves

## Workflow

The machine learning workflow followed these steps:

1. Data cleaning and preprocessing  
2. Encoding categorical variables  
3. Train-test split  
4. Feature scaling using StandardScaler  
5. Logistic Regression model training  
6. Feature analysis using model coefficients  
7. Model evaluation using classification metrics and confusion matrix  

## Model Performance

Accuracy: 0.74

Churn prediction metrics:

- Precision: 0.50
- Recall: 0.79
- F1 Score: 0.61

The model achieves a relatively high recall for churn prediction, meaning it successfully identifies most customers who are likely to leave.

## Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

telco_customer_churn/
│
├── Telco-Customer-Churn.ipynb
├── Telco-Customer-Churn.html
└── README.md


## Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Trying other models such as Random Forest or Gradient Boosting
- Feature engineering to improve predictive performance

## Project Structure
