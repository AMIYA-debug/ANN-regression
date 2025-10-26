# Customer Churn Prediction App

This is a **Streamlit web application** that predicts whether a customer is likely to **churn** (leave the bank) or **stay**, based on their profile information. The app uses a pre-trained **Artificial Neural Network (ANN)** model along with preprocessing using **Label Encoding**, **One-Hot Encoding**, and **scaling**.

---

## Features

- Predict churn for a **single customer**.
- Inputs for all relevant customer features:
  - Credit Score
  - Geography (France, Spain, Germany)
  - Gender (Male, Female)
  - Age
  - Tenure (Years)
  - Balance
  - Number of Products
  - Has Credit Card
  - Is Active Member
  - Estimated Salary
- Preprocessing:
  - **Label Encoding** for Gender
  - **One-Hot Encoding** for Geography
  - **Scaling** using saved scaler
- Outputs the **probability of churn** along with a message:
  - High probability → Customer likely to **churn**
  - Low probability → Customer likely to **stay**

---

## Requirements

- Python 3.x
- Streamlit
- TensorFlow / Keras
- scikit-learn
- pandas
- pickle

Install the required packages:

```bash
pip install streamlit pandas scikit-learn tensorflow
