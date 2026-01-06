import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# App title
st.title("🌲 Random Forest Classification Demo")

# Sample dataset (Loan Approval Prediction)
data = {
    "Income": [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000],
    "Credit_Score": [550, 580, 600, 620, 650, 680, 700, 720, 750, 780],
    "Loan_Approved": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

st.subheader("📊 Dataset")
st.write(df)

# Features and target
X = df[["Income", "Credit_Score"]]
y = df["Loan_Approved"]

# Train-test s
