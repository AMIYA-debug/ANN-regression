import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

with open('scaler.pkl','rb') as f:
   scaler= pickle.load(f)

with open('label.pkl','rb') as f:
   label=pickle.load(f)

with open('onehot.pkl','rb') as f:
   onehot=pickle.load(f)

model=load_model('ANN_model.keras')
st.title("Customer Churn Prediction")


credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.number_input("Age", min_value=18, max_value=100, value=40)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0, value=50000)
num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0, value=60000)


input_df = pd.DataFrame({
    'CreditScore':[credit_score],
    'Geography':[geography],
    'Gender':[gender],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})


input_df['Gender'] = label.transform(input_df['Gender'])


geo_encoded = onehot.transform(input_df[['Geography']]).toarray()

geo_df = pd.DataFrame(geo_encoded)

input_df = input_df.drop('Geography', axis=1)
input_df = pd.concat([geo_df, input_df], axis=1)
input_df.columns = input_df.columns.astype(str)

input_scaled = scaler.transform(input_df)


if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)
    churn_prob = prediction[0][0] if prediction.shape[1]==1 else prediction[0]

    if churn_prob >= 0.5:
        st.error(f"Customer is likely to **churn** (Probability: {churn_prob:.2f})")
    else:
        st.success(f"Customer is likely to **stay** (Probability: {churn_prob:.2f})")



