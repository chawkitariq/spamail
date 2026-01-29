# Spamail AWS Infrastructure

Deploy the Spamail email preprocessing pipeline to AWS using Terraform.

## What It Does

- **S3 Buckets**: Store raw emails, processed CSV, and ML data/artifacts
- **Lambda Functions**: Process emails and deploy SageMaker endpoints
- **ECR Repositories**: Host Lambda and SageMaker container images
- **SageMaker Pipeline**: Automated ML training and deployment
- **API Gateway**: REST API for spam classification predictions
- **CloudWatch**: Monitoring and logging for all services

## Quick Start

### 1. Prerequisites
- AWS CLI configured
- Terraform installed
- Docker installed

### 2. Deploy Infrastructure
```bash
cd terraform
terraform init \
  -backend-config="bucket=ct-terraform-state-backend" \
  -backend-config="key=spamail-terraform.tfstate" \
  -backend-config="region=eu-west-3"
terraform plan
terraform apply
```

### 3. Upload Emails and Trigger Processing
```bash
# Upload ham emails
aws s3 rsync ../datas/raw/ham/ s3://dev-spamail-bucket/raw/ham/ --recursive

# Trigger ham processing
aws s3 cp ../datas/raw/ham/_COMPLETE s3://dev-spamail-bucket/raw/ham/_COMPLETE

# Upload spam emails
aws s3 rsync ../datas/raw/spam/ s3://dev-spamail-bucket/raw/spam/ --recursive

# Trigger spam processing
aws s3 cp ../datas/raw/spam/_COMPLETE s3://dev-spamail-bucket/raw/spam/_COMPLETE
```

### 4. Download Results
```bash
aws s3 cp s3://dev-spamail-bucket/processed/email.csv ./
```

## How It Works

1. **Upload Emails** → Files go to `raw/ham/` or `raw/spam/`
2. **Upload _COMPLETE** → Upload `_COMPLETE` file to trigger processing
3. **Lambda Processes** → Processes ALL emails in that folder
4. **CSV Updates** → Appends cleaned emails to `processed/email.csv`
5. **Train Model** → SageMaker pipeline trains spam classifier
6. **Deploy Endpoint** → Model deployed as serverless endpoint
7. **Classify Emails** → API Gateway provides prediction API

## Architecture

```
Upload Emails → Upload _COMPLETE → Lambda → Process All → Append to CSV
                                      ↓
SageMaker Pipeline ← Train Model ← Deploy Endpoint ← API Gateway ← Predictions
```

- **Input**: Raw email files in `raw/ham/` or `raw/spam/`
- **Trigger**: `_COMPLETE` file uploaded to same folder
- **Processing**: Removes HTML, special chars, converts to lowercase
- **Output**: CSV with `text` and `label` columns (0=ham, 1=spam)
- **ML Pipeline**: Automated training with TF-IDF + MultinomialNB
- **Inference**: REST API for real-time spam classification

## SageMaker Components

### Training Pipeline
- **Container**: Custom training image with scikit-learn
- **Algorithm**: TF-IDF vectorization + MultinomialNB classifier
- **Input**: Processed email CSV from S3
- **Output**: Trained model artifacts saved to S3

### Inference Endpoint
- **Container**: Flask app with nginx/gunicorn
- **API**: `/ping` health check, `/invocations` for predictions
- **Input**: JSON with email text
- **Output**: Spam probability and classification

### API Gateway
- **Endpoints**: REST API for spam classification
- **Integration**: Serverless SageMaker endpoint
- **Authentication**: IAM-based (configurable)

## Clean Up

```bash
terraform destroy
```

## Notes

- Lambda processes **all emails in folder** when `_COMPLETE` is uploaded
- Upload `_COMPLETE` file to trigger processing for that folder
- Prevents infinite loops by ignoring `processed/` folder
- Uses Docker container for dependencies (pandas, etc.)
- Timeout: 5 minutes, Memory: 1GB
