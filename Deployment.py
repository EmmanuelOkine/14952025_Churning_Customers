import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

# Load the trained model
model = load_model('best_model.h5')  
label=pickle.load(open('label_encoder.pkl',rb))
scaler = pickle.load(open('scaler.pkl', 'rb'))  

# Function to preprocess input data
def preprocess_input(data):
    # Extract categorical columns
    categorical_columns = ['Contract', 'OnlineSecurity', 'OnlineBackup', 'TechSupport',
                           'PaymentMethod', 'gender', 'Partner', 'Dependents']

    # Extract categorical data
    data_categorical = data[categorical_columns]

    # One-hot encode categorical columns
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_data = encoder.fit_transform(data_categorical)
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names(categorical_columns))

    # Drop original categorical columns and concatenate encoded ones
    data = pd.concat([data.drop(categorical_columns, axis=1), encoded_df], axis=1)

    # Scale numerical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data[numerical_features] = scaler.transform(data[numerical_features])

    return data

def predict_churn(input_data):
    input_data = preprocess_input(input_data)
    # Make predictions
    predictions = model.predict(input_data)
    return predictions

# Streamlit app
def main():
    st.title('Customer Churn Prediction')

    st.write('Enter customer information to predict churn:')
    
    tenure = st.slider('Tenure (months)', min_value=0, max_value=100, value=50)
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=5000.0)
    
   
    contract_options = ['Month-to-month', 'One year', 'Two year']
    contract = st.selectbox('Contract', contract_options)

    online_security_options = ['No', 'Yes']
    online_security = st.selectbox('Online Security', online_security_options)

    online_backup_options = ['No', 'Yes']
    online_backup = st.selectbox('Online Backup', online_backup_options)

    tech_support_options = ['No', 'Yes']
    tech_support = st.selectbox('Tech Support', tech_support_options)

    payment_method_options = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    payment_method = st.selectbox('Payment Method', payment_method_options)

    gender_options = ['Male', 'Female']
    gender = st.selectbox('Gender', gender_options)

    partner_options = ['Yes', 'No']
    partner = st.selectbox('Partner', partner_options)

    dependents_options = ['Yes', 'No']
    dependents = st.selectbox('Dependents', dependents_options)

    if st.button('Predict'):
        user_input = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'Contract': [contract],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'TechSupport': [tech_support],
            'PaymentMethod': [payment_method],
            'gender': [gender],
            'Partner': [partner],
            'Dependents': [dependents]
        })

        prediction = predict_churn(user_input)
        churn_probability = prediction[0][0]
        churn_percentage = churn_probability * 100

        if churn_probability >= 0.5:
            st.error(f'Churn Prediction: Customer may churn with confidence level {churn_percentage:.2f}%')
        else:
            not_churn_percentage = (1 - churn_probability) * 100
            st.success(f'Churn Prediction: Customer may not churn with confidence level {not_churn_percentage:.2f}%')

if __name__ == '__main__':
    main()
