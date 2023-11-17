# 14952025_Churning_Customers
# Customer Churn Prediction Project

This project is dedicated to predicting customer churn in a telecommunications company using machine learning techniques. Customer churn, also known as customer attrition, is a critical metric for businesses, representing the percentage of customers who stop using a company's products or services during a specific time frame.

## Overview

The project aims to address the following key aspects:

1. **Problem Statement:** Predicting customer churn is essential for businesses to identify potential churners and take proactive measures to retain them. By analyzing customer data, we aim to build a predictive model that can anticipate whether a customer is likely to churn.

2. **Dataset:** The project utilizes a dataset containing various customer-related features, such as contract type, tenure, monthly charges, internet services, and more. The dataset was obtained from [source of dataset] and includes both numerical and categorical data.

3. **Project Structure:** The project comprises several steps, including data preprocessing, exploratory data analysis (EDA), model training using neural networks, hyperparameter tuning, and saving models and preprocessing objects for future use.

## Project Workflow

### 1. Data Preprocessing

- **Handling Missing Values:** Dealing with missing data by imputing or removing null values in the dataset.
- **Data Cleaning:** Ensuring consistency in the dataset by removing duplicates or irrelevant columns.
- **Encoding Categorical Variables:** Transforming categorical variables into a numerical format suitable for machine learning models.
- **Feature Scaling:** Scaling numerical features to a standard range to avoid bias in the model.

### 2. Exploratory Data Analysis (EDA)

- **Statistical Overview:** Descriptive statistics and data distribution of key features using visualizations such as histograms, count plots, and correlation matrices.
- **Understanding Patterns:** Identifying correlations, trends, and patterns in the data that might influence customer churn.

### 3. Model Training

- **Neural Network Model:** Utilizing the TensorFlow/Keras library to build a neural network model using the Functional API.
- **Model Evaluation:** Training the model, evaluating its performance on metrics like accuracy and AUC, and optimizing it for better predictions.

### 4. Hyperparameter Tuning

- **Grid Search:** Conducting a grid search to find the optimal hyperparameters for the neural network model, improving its predictive capability.

### 5. Model Saving

- **Saving Models:** Saving the trained neural network model, scaler used for feature scaling, and label encoder used for categorical variable encoding as pickle files for future use.

## Project Structure

- **customer_churn.ipynb:** Jupyter Notebook containing the Python code for data preprocessing, model training, and evaluation.
- **best_model.h5:** Trained neural network model saved in the HDF5 file format.
- **scaler.pkl:** Pickle file containing the scaler used for feature scaling.
- **label_encoder.pkl:** Pickle file containing the label encoder used for encoding categorical variables.
- **CustomerChurn_dataset.csv:** Input dataset containing customer-related information.

## Usage Guide

1. **Environment Setup:** Ensure the necessary Python libraries (Keras, TensorFlow, Scikit-learn, Pandas, etc.) are installed.
2. **Running the Notebook:** Execute the `customer_churn.ipynb` Jupyter Notebook to preprocess data, train the model, and save models.
3. **Exploring Results:** Analyze model metrics (accuracy, AUC, etc.) and use saved models to predict customer churn on new data.
4. Deployment Setup using Streamlit to deploy a web based applicationm for use .
