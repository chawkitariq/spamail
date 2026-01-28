# Spamail AWS Infrastructure

Deploy the Spamail email preprocessing pipeline to AWS using Terraform.

## What It Does

- **S3 Bucket**: Stores raw emails and processed CSV
- **Lambda Function**: Automatically processes uploaded emails
- **ECR Repository**: Hosts the Lambda container image

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

### 3. Upload Emails
```bash
# Upload ham emails
aws s3 rsync ../datas/raw/ham/ s3://spamail-bucket/raw/ham/ --recursive

# Upload spam emails
aws s3 rsync ../datas/raw/spam/ s3://spamail-bucket/raw/spam/ --recursive
```

### 4. Download Results
```bash
aws s3 cp s3://spamail-bucket/processed/email.csv ./
```

## How It Works

1. **Upload Email** → File goes to `raw/ham/` or `raw/spam/`
2. **Lambda Triggers** → Processes only that specific file
3. **CSV Updates** → Appends cleaned email to `processed/email.csv`
4. **No Loops** → Skips processed files to prevent infinite triggers

## Architecture

```
S3 Upload → Lambda → Clean Text → Append to CSV
```

- **Input**: Raw email files in `raw/ham/` or `raw/spam/`
- **Processing**: Removes HTML, special chars, converts to lowercase
- **Output**: CSV with `text` and `label` columns (0=ham, 1=spam)

## Clean Up

```bash
terraform destroy
```

## Notes

- Lambda processes **one file at a time** (not all files)
- Prevents infinite loops by ignoring `processed/` folder
- Uses Docker container for dependencies (pandas, etc.)
- Timeout: 5 minutes, Memory: 1GB
