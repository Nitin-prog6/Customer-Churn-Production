import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

st.title("📊 Customer Churn Prediction Dashboard")

uploaded_file = st.file_uploader("Upload Telco Churn Dataset", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue 👆")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Raw Data")
st.write(df.head())

# Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, drop_first=True)

# Split
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Prediction
y_prob = model.predict_proba(X_test)[:, 1]

threshold = st.slider("Select Threshold", 0.0, 1.0, 0.3)
y_pred = (y_prob > threshold).astype(int)

st.subheader("Predictions")
st.write(pd.DataFrame({
    "Probability": y_prob[:10],
    "Prediction": y_pred[:10]
}))

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

st.subheader("Model Performance")
st.write(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

# Business cost
cost_churn = 500
cost_retention = 50

loss = fn * cost_churn
campaign = (tp + fp) * cost_retention
total = loss + campaign

st.subheader("Business Impact")
st.write(f"Loss: ${loss}")
st.write(f"Campaign Cost: ${campaign}")
st.write(f"Total Cost: ${total}")

# Default predictions
y_pred_default = model.predict(X_test)

tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_test, y_pred_default).ravel()

loss_default = fn_d * cost_churn + (tp_d + fp_d) * cost_retention

improvement = loss_default - total
percent = round(improvement / loss_default * 100, 2)

st.subheader("📉 Cost Optimization")
st.write(f"Cost Reduction: ${improvement}")
st.write(f"Improvement: {percent}%")

st.success("✅ Successfully reduced business cost using ML optimization!")