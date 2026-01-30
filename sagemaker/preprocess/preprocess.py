"""
SageMaker Processing script to preprocess raw emails into CSV format.
"""
import argparse
import pandas as pd
import re
from pathlib import Path


def clean_text(text):
    """Clean email text by removing HTML tags and special characters."""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower().strip()


def read_email_file(file_path):
    """Read and clean email content from local file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    return clean_text(content)


def process_folder_emails(folder_path, label):
    """Process all email files in a local folder."""
    emails = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return emails
    
    for file_path in folder.rglob('*'):
        if file_path.is_file() and not file_path.name.startswith('_'):
            try:
                cleaned_text = read_email_file(file_path)
                if cleaned_text:
                    emails.append({'text': cleaned_text, 'label': label})
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return emails


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    args = parser.parse_args()

    # Process ham emails
    ham_path = Path(args.input_dir) / 'ham'
    ham_emails = process_folder_emails(str(ham_path), 0)
    
    # Process spam emails
    spam_path = Path(args.input_dir) / 'spam'
    spam_emails = process_folder_emails(str(spam_path), 1)
    
    # Combine and save
    all_emails = ham_emails + spam_emails
    if not all_emails:
        raise ValueError("No emails found to process")
    
    df = pd.DataFrame(all_emails)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to local output directory
    output_file = output_path / 'email.csv'
    df.to_csv(output_file, index=False)
    
if __name__ == '__main__':
    main()
