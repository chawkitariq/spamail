import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "datas", "processed", "email.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models")

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].tolist()

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Vectorize text
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    print("Evaluation Report:")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_PATH, "spam_classifier.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_PATH, "vectorizer.pkl"))

if __name__ == "__main__":
    main()
