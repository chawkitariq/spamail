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


def lambda_handler(event, context):
    """
    Process a single email file triggered by S3 upload and append to CSV.
    Prevents infinite loops by only processing files in raw/ham/ or raw/spam/.
    """
    try:
        bucket_name = os.environ.get('BUCKET_NAME')
        
        # Get the uploaded file details from S3 event
        triggered_key = event['Records'][0]['s3']['object']['key']
        print(f"Processing: {triggered_key}")
        
        # Skip non-email files and prevent infinite loops
        if triggered_key.endswith(('.gitkeep', '/')) or triggered_key.startswith('processed/'):
            return {'statusCode': 200, 'body': json.dumps({'message': 'Skipped'})}
        
        # Determine if email is ham (0) or spam (1)
        if 'raw/ham/' in triggered_key:
            label = 0
        elif 'raw/spam/' in triggered_key:
            label = 1
        else:
            return {'statusCode': 400, 'body': json.dumps({'message': 'Invalid folder'})}
        
        # Read and clean the email content
        file_obj = s3_client.get_object(Bucket=bucket_name, Key=triggered_key)
        content = file_obj['Body'].read().decode('utf-8', errors='ignore')
        cleaned_text = clean_text(content)
        
        # Load existing CSV or create new one
        output_key = 'processed/email.csv'
        try:
            csv_obj = s3_client.get_object(Bucket=bucket_name, Key=output_key)
            df = pd.read_csv(io.BytesIO(csv_obj['Body'].read()))
        except s3_client.exceptions.NoSuchKey:
            df = pd.DataFrame(columns=['text', 'label'])
        
        # Append new email to DataFrame
        df = pd.concat([df, pd.DataFrame([{'text': cleaned_text, 'label': label}])], ignore_index=True)
        
        # Save updated CSV back to S3
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=output_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        print(f"Success! Total emails in CSV: {len(df)}")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Email processed',
                'file': triggered_key,
                'total_emails': len(df)
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
