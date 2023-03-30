import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Title of the web app
st.title("AutoML Web App")

# Upload CSV data file
uploaded_file = st.file_uploader("Upload CSV data file", type=["csv"])

if uploaded_file is not None:
    # Read CSV file
    data = pd.read_csv(uploaded_file)
    # Display data
    st.write("Data:")
    st.write(data)

    # Select target variable
    target_variable = st.selectbox("Select target variable", data.columns)

    # Select Machine Learning model
    models = {
        "Random Forest": RandomForestRegressor()
    }
    model_name = st.selectbox("Select a model", list(models.keys()))

    # Train model
    model = models[model_name]
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    model.fit(X, y)

    # Make predictions
    st.write("Enter feature values for prediction:")
    features = {}
    for feature in X.columns:
        feature_value = st.number_input(feature, step=0.01)
        features[feature] = feature_value
    input_data = pd.DataFrame(features, index=[0])
    prediction = model.predict(input_data)
    st.write("Prediction:", prediction)

    # Allow user to adjust inputs and hyperparameters
    st.sidebar.write("## Model Hyperparameters")
    n_estimators = st.sidebar.slider("Number of estimators", 1, 100, 10)
    max_depth = st.sidebar.slider("Max depth", 1, 20, 5)

    # Display results
    st.write("### Results")
    st.write(f"Target variable: {target_variable}")
    st.write(f"Number of estimators: {n_estimators}")
    st.write(f"Max depth: {max_depth}")
    st.write(f"Prediction: {prediction}")
