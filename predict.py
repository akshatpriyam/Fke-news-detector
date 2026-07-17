import joblib

# Load saved model
model = joblib.load("model/fake_news_model.pkl")

# Load saved TF-IDF vectorizer
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

# User input
news = input("Enter News: ")

# Convert input into TF-IDF features
news_vector = tfidf.transform([news])

# Predict
prediction = model.predict(news_vector)

# Display result
if prediction[0] == 1:
    print("\n✅ REAL NEWS")
else:
    print("\n❌ FAKE NEWS")