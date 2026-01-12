import os
import glob
import pandas as pd
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_PATH = os.path.join(PROJECT_ROOT, "datas", "raw")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "datas", "processed")

def clean_text(text):
    # Basic cleaning: remove HTML tags, non-alphabetic chars, lowercase
    text = re.sub(r"<.*?>", " ", text)       # remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text) # keep only letters and spaces
    text = text.lower().strip()
    return text

def process_folder(folder, label):
    files = glob.glob(os.path.join(RAW_PATH, folder, "*"))
    emails = []
    for f in files:
        with open(f, encoding="utf-8", errors="ignore") as file:
            content = file.read()
            emails.append(clean_text(content))
    df = pd.DataFrame({"text": emails, "label": label})
    return df

# Process ham and spam
ham_df = process_folder("ham", 0)   # label 0 for ham
spam_df = process_folder("spam", 1) # label 1 for spam

# Merge into one dataset
email_df = pd.concat([ham_df, spam_df], ignore_index=True)

# Save to CSV
os.makedirs(PROCESSED_PATH, exist_ok=True)
email_df.to_csv(os.path.join(PROCESSED_PATH, "email.csv"), index=False)

print("✅ Processing complete. File saved in datas/processed/email.csv")
