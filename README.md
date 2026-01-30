# Spamail – Spam Email Classifier

## Overview
Spamail is a machine learning pipeline for spam email classification with local development capabilities and AWS cloud deployment.

**Local Development:**
- Preprocess raw email files into structured CSV data
- Train spam classification models using scikit-learn
- Test models with inference scripts

**Key Features:**
- TF-IDF vectorization + Multinomial Naive Bayes classification
- Docker containerized training and inference
- REST API for real-time predictions

---

## Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/chawkitariq/spamail.git
   cd spamail
   ```

2. **Create virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Preprocessing
Run the preprocessing script to convert raw emails into a structured CSV:

```bash
python3 src/preprocess.py
```

This generates:

- `datas/processed/email.csv` → with two columns:
  - `text`: plain text content of the email
  - `label`: `0` = ham, `1` = spam

---

## Training
Train the spam classifier:

```bash
python3 src/train.py
```

This will:
- Load `email.csv`
- Vectorize text using **TF‑IDF**
- Train a **Naive Bayes** (or Logistic Regression) model
- Print evaluation metrics (precision, recall, F1‑score, accuracy)
- Save artifacts in `models/`

---

## Inference
Test the model on sample emails:

```bash
python3 src/inference.py
```

Example output:
```
Email: Congratulations! You have won a free prize, click here to claim.
 → Prediction: SPAM

Email: Hi team, please find the meeting notes attached.
 → Prediction: HAM

Email: Cheap meds available now, limited offer!
 → Prediction: SPAM

Email: Looking forward to our lunch tomorrow.
 → Prediction: HAM

Email: Get paid to work from home, sign up today!
 → Prediction: SPAM

Email: Don't forget to submit your project report by Friday.
 → Prediction: HAM

Email: Exclusive deal just for you, act fast!
 → Prediction: SPAM

Email: Can we reschedule our appointment to next week?
 → Prediction: HAM
```

---

## AWS Deployment

For detailed AWS deployment instructions, see [terraform/README.md](terraform/README.md)
