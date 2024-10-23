import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Streamlit App
st.title("Heart Disease Prediction App")

# 1. Upload dataset
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded")
    
    # 2. Preprocess Data
    label_encoder = LabelEncoder()
    df['Heart Disease'] = label_encoder.fit_transform(df['Heart Disease'])  # Convert 'Absence'/'Presence' to 0/1
    X = df.drop(columns='Heart Disease')
    y = df['Heart Disease']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train the Logistic Regression model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    
    # 4. Display accuracy on the test set
    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.write(f"Logistic Regression Model Accuracy: {accuracy:.2f}")
    
    # 5. Prediction
    st.header("Make a Prediction")
    st.write("Input the values for the following features:")
    
    # Collect user input for all features
    input_data = {}
    for col in X.columns:
        value = st.number_input(f"Input for {col}", value=float(X[col].mean()))
        input_data[col] = value
    
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Predict the outcome
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # Display prediction
    st.write(f"Prediction (0 = Absence, 1 = Presence): {prediction[0]}")
    st.write(f"Probability of Heart Disease Presence: {prediction_proba[0][1]:.2f}")
