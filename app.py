import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st



# load pickled preprocessors (scaler, label encoder, one-hot encoder)
if not os.path.exists('scaler.pkl') or not os.path.exists('label.pkl') or not os.path.exists('onehot.pkl'):
    raise FileNotFoundError("Required preprocessing files not found. Make sure 'scaler.pkl', 'label.pkl' and 'onehot.pkl' exist in the working directory.")

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

with open('label.pkl','rb') as f:
    label = pickle.load(f)

with open('onehot.pkl','rb') as f:
    onehot = pickle.load(f)

model = load_model('ANN_model.keras')


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
input_df = pd.concat([input_df, geo_df], axis=1)


input_df.columns = input_df.columns.astype(str)


feature_columns = ['CreditScore','Gender','Age','Tenure','Balance','NumOfProducts',
                   'HasCrCard','IsActiveMember','EstimatedSalary','0','1','2']
input_df = input_df[feature_columns]


# Try to transform the input. If there's a shape mismatch, attempt to reconstruct
# the training features from the original CSV, fit a compatible scaler, and retry.
try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    # common cause: mismatch in number of features
    st.warning(f"Scaler transform failed: {e}. Attempting to rebuild scaler from training data...")

    # Rebuild feature matrix the same way as in training notebook
    try:
        df_train = pd.read_csv('Churn_Modelling.csv')
        # replicate preprocessing from training
        df_train.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True)
        # label encoder and onehot should be the same objects we loaded
        df_train['Gender'] = label.transform(df_train['Gender'])
        geo = onehot.transform(df_train[['Geography']])
        geo_df = pd.DataFrame(geo.toarray())
        df_train = pd.concat([df_train, geo_df], axis=1)
        df_train.drop(['Geography'], axis=1, inplace=True)

        X_train = df_train.drop('Exited', axis=1)
        X_train.columns = X_train.columns.astype(str)

        # fit a new scaler on the reconstructed training features
        new_scaler = StandardScaler()
        new_scaler.fit(X_train)

        # save the newly fitted scaler (overwrite existing scaler.pkl)
        with open('scaler.pkl','wb') as f:
            pickle.dump(new_scaler, f)

        scaler = new_scaler

        # Align input_df columns to training columns
        missing_cols = [c for c in X_train.columns if c not in input_df.columns]
        if missing_cols:
            raise ValueError(f"Input is missing columns that were present during training: {missing_cols}")

        input_df = input_df.reindex(columns=X_train.columns)
        input_scaled = scaler.transform(input_df)
        st.success("Rebuilt scaler from training data and transformed input successfully.")
    except Exception as e2:
        st.error(f"Failed to rebuild scaler and transform input: {e2}")
        raise

if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)
    churn_prob = prediction[0][0] if prediction.shape[1]==1 else prediction[0]

    if churn_prob >= 0.5:
        st.error(f"Customer is likely to **churn** (Probability: {churn_prob:.2f})")
    else:
        st.success(f"Customer is likely to **stay** (Probability: {churn_prob:.2f})")



