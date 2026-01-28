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


def process_emails(bucket_name, prefix, label):
    """
    Process all emails from a specific folder (ham or spam).
    
    Args:
        bucket_name: S3 bucket name
        prefix: Folder path (e.g., 'raw/ham/')
        label: 0 for ham, 1 for spam
    
    Returns:
        List of email dictionaries with text and label
    """
    emails = []
    email_type = "ham" if label == 0 else "spam"
    
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    if 'Contents' not in response:
        print(f"No files found in {prefix}")
        return emails
    
    for obj in response['Contents']:
        key = obj['Key']
        
        # Skip directories and .gitkeep files
        if key.endswith('/') or key.endswith('.gitkeep'):
            continue
        
        try:
            # Download and clean the file
            file_obj = s3_client.get_object(Bucket=bucket_name, Key=key)
            content = file_obj['Body'].read().decode('utf-8', errors='ignore')
            cleaned_text = clean_text(content)
            
            emails.append({'text': cleaned_text, 'label': label})
            print(f"Processed: {key}")
            
        except Exception as e:
            print(f"Failed to process {key}: {e}")
            continue
    
    print(f"Completed: {len(emails)} {email_type} emails processed")
    return emails


def lambda_handler(event, context):
    """
    Lambda function to preprocess spam/ham emails.
    Triggered by S3 uploads, processes all files in raw/ folders,
    and creates a CSV file in the processed/ folder.
    """
    try:
        bucket_name = os.environ.get('BUCKET_NAME')
        if not bucket_name:
            raise ValueError("BUCKET_NAME environment variable not set")
        
        ham_emails = process_emails(bucket_name, 'raw/ham/', label=0)
        spam_emails = process_emails(bucket_name, 'raw/spam/', label=1)
        
        all_emails = ham_emails + spam_emails
        
        if not all_emails:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'No emails found to process',
                })
            }
        
        df = pd.DataFrame(all_emails)
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        output_key = 'processed/email.csv'
        s3_client.put_object(
            Bucket=bucket_name,
            Key=output_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        print(f"Successfully created {output_key}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processing complete'
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': "Internal server error",
            })
        }
