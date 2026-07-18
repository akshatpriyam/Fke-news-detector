import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

st.title("📰 Fake News Detector")

st.write("Enter a news article or headline below.")

news = st.text_area("News")

if st.button("Check News"):
    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        news_vector = tfidf.transform([news])
        prediction = model.predict(news_vector)

        if prediction[0] == 1:
            st.error("❌ This looks like FAKE NEWS")
        else:
            st.success("✅ This looks like REAL NEWS")