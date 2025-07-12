import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("ticket_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit app
st.title("🎫 IT Ticket Category Classifier")

ticket_text = st.text_area("Enter the ticket description below:")

if st.button("Predict Category"):
    if ticket_text.strip() == "":
        st.warning("Please enter a ticket description.")
    else:
        vec = vectorizer.transform([ticket_text])
        prediction = model.predict(vec)[0]
        st.success(f"Predicted Ticket Category: **{prediction}**")
