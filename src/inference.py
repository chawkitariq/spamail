import os
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "spam_classifier.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "vectorizer.pkl")

def main():
    # Load artifacts
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Example emails to test
    samples = [
        "Congratulations! You have won a free prize, click here to claim.",
        "Hi team, please find the meeting notes attached.",
        "Cheap meds available now, limited offer!",
        "Looking forward to our lunch tomorrow.",
        "Get paid to work from home, sign up today!",
        "Don't forget to submit your project report by Friday.",
        "Exclusive deal just for you, act fast!",
        "Can we reschedule our appointment to next week?"
    ]

    # Transform and predict
    X = vectorizer.transform(samples)
    preds = model.predict(X)

    # Show results
    for text, label in zip(samples, preds):
        print(f"Email: {text}\n → Prediction: {'SPAM' if label == 1 else 'HAM'}\n")

if __name__ == "__main__":
    main()
