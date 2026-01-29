import json
import boto3
import pandas as pd
import re
import io
import os

s3_client = boto3.client('s3')


def clean_text(text):
    """Clean email text by removing HTML tags and special characters."""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower().strip()


def should_skip_file(file_key):
    """Check if file should be skipped."""
    return file_key.endswith(('_COMPLETE', '/'))


def read_email_from_s3(bucket_name, file_key):
    """Read and clean email content from S3."""
    file_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    content = file_obj['Body'].read().decode('utf-8', errors='ignore')
    return clean_text(content)


def load_csv_from_s3(bucket_name, csv_key):
    """Load existing CSV from S3 or return empty DataFrame."""
    try:
        csv_obj = s3_client.get_object(Bucket=bucket_name, Key=csv_key)
        return pd.read_csv(io.BytesIO(csv_obj['Body'].read()))
    except s3_client.exceptions.NoSuchKey:
        return pd.DataFrame(columns=['text', 'label'])


def save_csv_to_s3(bucket_name, csv_key, dataframe):
    """Save DataFrame as CSV to S3."""
    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, index=False)
    s3_client.put_object(
        Bucket=bucket_name,
        Key=csv_key,
        Body=csv_buffer.getvalue(),
        ContentType='text/csv'
    )


def get_folder_info(triggered_key):
    """Determine folder to process from trigger key."""
    if 'raw/ham/' in triggered_key:
        return 'raw/ham/', 0, 'ham'
    elif 'raw/spam/' in triggered_key:
        return 'raw/spam/', 1, 'spam'
    return None, None, None


def process_folder_emails(bucket_name, folder_prefix, label):
    """Process all emails in a folder."""
    emails = []
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_prefix)
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                if not should_skip_file(file_key):
                    cleaned_text = read_email_from_s3(bucket_name, file_key)
                    emails.append({'text': cleaned_text, 'label': label})
    
    return emails


def lambda_handler(event, context):
    """Process emails when _COMPLETE is uploaded."""
    try:
        bucket_name = os.environ.get('BUCKET_NAME')
        triggered_key = event['Records'][0]['s3']['object']['key']
        print(f"Triggered by: {triggered_key}")
        
        if not triggered_key.endswith('_COMPLETE'):
            return {
                'statusCode': 200, 
                'body': json.dumps({
                    'message': 'Not _COMPLETE file'
                })
            }
        
        folder_prefix, label, folder_name = get_folder_info(triggered_key)
        if not folder_prefix:
            return {
                'statusCode': 400, 
                'body': json.dumps({
                    'message': 'Invalid folder'
                })
            }
        
        print(f"Processing {folder_name} emails from {folder_prefix}")
        
        # Load existing CSV and process new emails
        output_key = 'processed/email.csv'
        existing_df = load_csv_from_s3(bucket_name, output_key)
        new_emails = process_folder_emails(bucket_name, folder_prefix, label)
        
        # Append and save
        combined_df = pd.concat([existing_df, pd.DataFrame(new_emails)], ignore_index=True)
        save_csv_to_s3(bucket_name, output_key, combined_df)
        
        print(f"Success! Added {len(new_emails)} {folder_name} emails. Total: {len(combined_df)}")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Processed {folder_name} emails',
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500, 
            'body': json.dumps({
                'message': 'Error occurred'
            })
        }