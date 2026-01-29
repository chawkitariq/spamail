import json
import boto3
import pandas as pd
import re
import io
import os

s3_client = boto3.client('s3')

def clean_text(text):
    """Clean email text by removing HTML tags and special characters."""
    text = re.sub(r"<.*?>", " ", text)       # Remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text) # Keep only letters and spaces
    return text.lower().strip()


def should_skip_file(file_key):
    """Check if file should be skipped (directories, metadata, or processed files)."""
    return file_key.endswith(('.gitkeep', '_COMPLETE', '/')) or file_key.startswith('processed/')


def get_email_label(file_key):
    """
    Determine email label from file path.
    Returns 0 for ham, 1 for spam, None for invalid paths.
    """
    if 'raw/ham/' in file_key:
        return 0
    elif 'raw/spam/' in file_key:
        return 1
    return None


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


def lambda_handler(event, context):
    """
    Process a single email file triggered by S3 upload and append to CSV.
    Prevents infinite loops by only processing files in raw/ham/ or raw/spam/.
    """
    try:
        bucket_name = os.environ.get('BUCKET_NAME')
        triggered_key = event['Records'][0]['s3']['object']['key']
        print(f"Processing: {triggered_key}")
        
        # Validate file
        if should_skip_file(triggered_key):
            return {
                'statusCode': 200, 
                'body': json.dumps({
                    'message': 'Skipped'
                })
            }
        
        label = get_email_label(triggered_key)
        if label is None:
            return {
                'statusCode': 400, 
                'body': json.dumps({
                    'message': 'Invalid folder'
                })
            }
        
        # Process email
        cleaned_text = read_email_from_s3(bucket_name, triggered_key)
        
        # Update CSV
        output_key = 'processed/email.csv'
        df = load_csv_from_s3(bucket_name, output_key)
        df = pd.concat([df, pd.DataFrame([{'text': cleaned_text, 'label': label}])], ignore_index=True)
        save_csv_to_s3(bucket_name, output_key, df)
        
        print(f"Success! Total emails in CSV: {len(df)}")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Email processed',
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500, 
            'body': json.dumps({
                'message': 'Internal server error'
            })
        }