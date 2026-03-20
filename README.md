# Customer-Churn-Production
End-to-end customer churn prediction project using machine learning -Logistic Regression, Random Forest, XGBoost with EDA, feature engineering and business insights visualization.
# Customer Churn Prediction

## Overview
This project predicts customer churn using machine learning models and provides business insights to reduce customer attrition.

## Dataset
Telco Customer Churn dataset containing customer demographics, services, and billing information.

## Workflow
- Data Cleaning (handled missing values, type conversion)
- Exploratory Data Analysis (EDA)
- Feature Engineering (one-hot encoding)
- Model Training (Logistic Regression, Random Forest, XGBoost)
- Model Evaluation (Accuracy, ROC-AUC, Precision, Recall)

## Results
- Logistic Regression performed best with ~78% accuracy and 0.70 ROC-AUC
- Moderate recall (~52%) for churn class indicates improvement opportunities

## Key Insights
- Month-to-month contracts have highest churn risk
- High monthly charges increase churn probability
- Low tenure customers are more likely to churn
- Long-term contracts improve retention

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

## Future Improvements
- Improve recall using SMOTE or class balancing
- Hyperparameter tuning
- Deployment using Streamlit or Flask
