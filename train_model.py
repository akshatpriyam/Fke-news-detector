import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
# Read the dataset
df = pd.read_csv("dataset/WELFake_Dataset.csv")
print(df["label"].value_counts())

print(df[["title", "label"]].head(10))

# Show original shape
print("Original Shape:", df.shape)

# Remove the extra column
df = df.drop(columns=["Unnamed: 0"])

# Remove missing values
df = df.dropna()

# Combine title and text into one column
df["content"] = df["title"] + " " + df["text"]

# Keep only the columns we need
df = df[["content", "label"]]

# Show cleaned shape
print("Cleaned Shape:", df.shape)

print(df["label"].value_counts())

# Display first 5 rows
print("\nFirst 5 rows after cleaning:\n")
print(df.head())
# Convert text into numbers using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

X = tfidf.fit_transform(df["content"])

y = df["label"]

print("\nTF-IDF Matrix Shape:", X.shape)

print("\nLabels Shape:", y.shape)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Create the model
model = RandomForestClassifier(random_state=42)

print("\nTraining the model... Please wait.")

# Train the model
model.fit(X_train, y_train)

print("Model trained successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Save the trained model
joblib.dump(model, "model/fake_news_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("\nModel and TF-IDF Vectorizer saved successfully!")
