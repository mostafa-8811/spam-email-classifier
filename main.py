import os
import re
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Ensure NLTK resources are available
nltk.download("stopwords", quiet=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """Clean, lowercase, remove stopwords, and stem input text."""
    text = re.sub(r"\W", " ", str(text))
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)


def train_and_evaluate(data_path: str = "data/spam.csv"):
    """Train the TF-IDF + Logistic Regression pipeline and save artifacts."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Please place spam.csv in the data/ directory."
        )

    print("Loading and processing dataset...")
    df = pd.read_csv(data_path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["cleaned_message"] = df["message"].apply(preprocess_text)

    # Feature Extraction
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df["cleaned_message"])
    y = df["label"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=54
    )

    # Model Training
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save trained artifacts for future inference
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/spam_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("Model and vectorizer saved to 'models/' directory.")

    return model, vectorizer


def predict_email(email_text: str, model=None, vectorizer=None) -> str:
    """Predict whether a given string is Spam or Not Spam."""
    if model is None or vectorizer is None:
        model = joblib.load("models/spam_model.pkl")
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    processed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0] == 1 else "Not Spam"


if __name__ == "__main__":
    # 1. Train model and display metrics
    model, vectorizer = train_and_evaluate("data/spam.csv")

    # 2. Test with sample inputs
    test_email = (
        "Congratulations! You've won a free iPhone. Click here to claim now."
    )
    result = predict_email(test_email, model, vectorizer)

    print(f"\nSample email: {test_email}")
    print(f"Prediction: {result}")
  
    
