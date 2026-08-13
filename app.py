import streamlit as st
import joblib

# Load trained model and TF-IDF vectorizer
model = joblib.load("fake_news_svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Fake News Credibility Analyzer",
    page_icon="📰",
    layout="centered"
)

# Title
st.title("📰 Fake News Credibility Analyzer")
st.write("Enter a news article or headline to check whether it is likely to be real or fake.")

# Text input
news_text = st.text_area(
    "Enter News Text",
    height=200,
    placeholder="Paste a news article or headline here..."
)

# Analyze button
if st.button("🔍 Analyze News"):

    if news_text.strip() == "":
        st.warning("Please enter some news text.")

    else:
        # Convert text into TF-IDF features
        news_tfidf = tfidf.transform([news_text])

        # Make prediction
        prediction = model.predict(news_tfidf)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 FAKE NEWS")
            st.write("The model predicts that this news is likely to be fake.")
        else:
            st.success("✅ REAL NEWS")
            st.write("The model predicts that this news is likely to be real.")

        st.caption("Prediction generated using TF-IDF and Linear SVM.")