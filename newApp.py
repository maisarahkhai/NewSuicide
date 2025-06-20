import streamlit as st
import joblib
import numpy as np

# Load trained model and TF-IDF vectorizer
model = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app interface
st.title("Suicide Detection Classifier")
st.write("Enter a Reddit post text to classify it as suicidal, depressive, or normal.")

# Text input
user_input = st.text_area("Reddit Post Text", "")

# When the user clicks the predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input with TF-IDF
        input_vector = vectorizer.transform([user_input])

        # Predict with trained model
        prediction = model.predict(input_vector)[0]

        st.success(f"Prediction: **{prediction}**")
