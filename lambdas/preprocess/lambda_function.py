import json
import boto3
import pandas as pd
import re
import io
import os
from datetime import datetime

s3_client = boto3.client('s3')

def clean_text(text):
    """Clean email text by removing HTML tags and non-alphabetic characters."""
    text = re.sub(r"<.*?>", " ", text)       # remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text) # keep only letters and spaces
    text = text.lower().strip()
    return text

def lambda_handler(event, context):
    """
    Lambda function triggered by S3 uploads to raw/ folder.
    Processes all files in raw/ham and raw/spam, then creates email.csv in processed/.
    """
    try:
        bucket_name = os.environ.get('BUCKET_NAME')
        
        # Log the triggering event
        print(f"Event received: {json.dumps(event)}")
        
        # Get the uploaded file details
        for record in event['Records']:
            uploaded_key = record['s3']['object']['key']
            print(f"New file uploaded: {uploaded_key}")
        
        # Process all files in raw/ham and raw/spam
        emails = []
        
        # Process ham emails (label = 0)
        print("Processing ham emails...")
        ham_objects = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix='raw/ham/'
        )
        
        if 'Contents' in ham_objects:
            for obj in ham_objects['Contents']:
                key = obj['Key']
                # Skip directory markers and .gitkeep files
                if key.endswith('/') or key.endswith('.gitkeep'):
                    continue
                    
                # Download and process file
                response = s3_client.get_object(Bucket=bucket_name, Key=key)
                content = response['Body'].read().decode('utf-8', errors='ignore')
                cleaned_text = clean_text(content)
                emails.append({'text': cleaned_text, 'label': 0})
                print(f"Processed: {key}")
        
        # Process spam emails (label = 1)
        print("Processing spam emails...")
        spam_objects = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix='raw/spam/'
        )
        
        if 'Contents' in spam_objects:
            for obj in spam_objects['Contents']:
                key = obj['Key']
                # Skip directory markers and .gitkeep files
                if key.endswith('/') or key.endswith('.gitkeep'):
                    continue
                    
                # Download and process file
                response = s3_client.get_object(Bucket=bucket_name, Key=key)
                content = response['Body'].read().decode('utf-8', errors='ignore')
                cleaned_text = clean_text(content)
                emails.append({'text': cleaned_text, 'label': 1})
                print(f"Processed: {key}")
        
        # Create DataFrame
        if not emails:
            print("No emails found to process")
            return {
                'statusCode': 200,
                'body': json.dumps('No emails to process')
            }
        
        df = pd.DataFrame(emails)
        print(f"Total emails processed: {len(df)}")
        print(f"Ham: {len(df[df['label']==0])}, Spam: {len(df[df['label']==1])}")
        
        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        # Upload to S3
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
                'message': 'Processing complete',
                'total_emails': len(df),
                'ham_count': int(len(df[df['label']==0])),
                'spam_count': int(len(df[df['label']==1])),
                'output_file': output_key,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
