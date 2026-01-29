import os
import argparse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    args = parser.parse_args()

    # Load dataset
    data_path = os.path.join(args.train, 'email.csv')
    df = pd.read_csv(data_path)
    
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
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    # Save model and vectorizer
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    joblib.dump(vectorizer, os.path.join(args.model_dir, 'vectorizer.joblib'))
    
    print(f"Model saved to {args.model_dir}")

if __name__ == "__main__":
    main()
