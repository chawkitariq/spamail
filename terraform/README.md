# Spamail AWS Infrastructure

Deploy the Spamail email preprocessing pipeline to AWS using Terraform.

## What It Does

![Spamail Schema](../docs/spamail.png)

- **S3 Buckets**: Store raw emails, processed CSV, and ML data/artifacts
- **ECR Repositories**: Host SageMaker container images
- **SageMaker Pipeline**: Automated ML training with preprocessing step
- **API Gateway**: REST API for spam classification predictions
- **Route 53**: Custom domain for API endpoints
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

### 3. Upload Emails and Start Pipeline
```bash
# Upload ham emails
aws s3 rsync ../datas/raw/ham/ s3://dev-spamail-bucket/raw/ham/ --recursive

# Upload spam emails
aws s3 rsync ../datas/raw/spam/ s3://dev-spamail-bucket/raw/spam/ --recursive

# Start SageMaker pipeline (includes preprocessing)
aws sagemaker start-pipeline-execution \
  --pipeline-name dev-spamail-pipeline \
  --region eu-west-3
```

## How It Works

1. **Upload Emails** → Files go to `raw/ham/` or `raw/spam/`
2. **Start Pipeline** → Execute SageMaker pipeline
3. **Preprocess Emails** → SageMaker Processing step cleans emails and creates CSV
4. **Train Model** → SageMaker training builds spam classifier
5. **Register Model** → Model saved to SageMaker Model Registry
6. **Deploy Endpoint** → Serverless inference endpoint created
7. **Classify Emails** → API Gateway provides prediction API with custom domain

## Architecture

```
Upload Emails → Start Pipeline → SageMaker Processing → Train Model → Deploy Endpoint → API Gateway → Custom Domain
```

- **Input**: Raw email files in `raw/ham/` or `raw/spam/`
- **Processing**: SageMaker Processing step removes HTML, special chars, converts to lowercase
- **Output**: CSV with `text` and `label` columns (0=ham, 1=spam)
- **ML Pipeline**: Automated training with TF-IDF + MultinomialNB
- **Inference**: REST API for real-time spam classification with custom domain

## SageMaker Components

### ML Pipeline
- **PreprocessEmails**: Batch processing of raw emails to CSV
- **TrainModel**: Trains the spam classifier on processed data
- **RegisterModel**: Registers the model in the model registry
- **CreateModel**: Creates a SageMaker model
- **DeployEndpoint**: Deploys a serverless endpoint

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

### API Gateway & Custom Domain
- **Endpoints**: REST API for spam classification
- **Integration**: Serverless SageMaker endpoint
- **Domain**: Custom domain via Route 53
- **Authentication**: IAM-based (configurable)

## Clean Up

```bash
terraform destroy
```
