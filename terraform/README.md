# Spamail Terraform Infrastructure

This directory contains Terraform configurations for deploying the Spamail infrastructure on AWS.

## Architecture

- **S3 Bucket** (`spamail-bucket`): Storage for email data
  - `raw/ham/`: Ham (non-spam) email files
  - `raw/spam/`: Spam email files
  - `processed/`: Generated `email.csv` file
  
- **Lambda Function** (`spamail-preprocess`): Triggered when new files are uploaded to `raw/` folder
  - Processes all emails in `raw/ham/` and `raw/spam/`
  - Generates consolidated `email.csv` in `processed/` folder

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Terraform installed (>= 1.0)
3. Python 3.11

## Setup Steps

### 1. Build Lambda Package

```bash
cd ../lambdas/preprocess
chmod +x build.sh
./build.sh
```

### 2. Create Pandas Layer (Optional)

For pandas support, you have two options:

**Option A: Use AWS-managed layer (Recommended)**

Update `lambda.tf` to use AWS's managed pandas layer or a public layer from Klayers:
```hcl
# Replace the pandas_layer resource with:
data "aws_lambda_layer_version" "pandas" {
  layer_name = "AWSSDKPandas-Python311"  # AWS managed layer
}

# Then use: layers = [data.aws_lambda_layer_version.pandas.arn]
```

**Option B: Create custom layer**

```bash
mkdir -p lambdas/layers/python
pip install pandas -t lambdas/layers/python/
cd lambdas/layers
zip -r pandas_layer.zip python/
cd ../../terraform
```

### 3. Initialize Terraform

```bash
cd terraform
terraform init
```

### 4. Plan Deployment

```bash
terraform plan
```

### 5. Apply Configuration

```bash
terraform apply
```

Review the plan and type `yes` to confirm.

## Usage

### Upload Test Files

```bash
# Upload ham emails
aws s3 cp ../datas/raw/ham/ s3://spamail-bucket/raw/ham/ --recursive

# Upload spam emails
aws s3 cp ../datas/raw/spam/ s3://spamail-bucket/raw/spam/ --recursive
```

The Lambda function will automatically trigger and create `processed/email.csv`.

### Download Processed File

```bash
aws s3 cp s3://spamail-bucket/processed/email.csv ./
```

### Monitor Lambda Logs

```bash
aws logs tail /aws/lambda/spamail-preprocess --follow
```

## Clean Up

To destroy all resources:

```bash
terraform destroy
```

## Variables

- `aws_region`: AWS region (default: `us-east-1`)
- `bucket_name`: S3 bucket name (default: `spamail-bucket`)

Override with:
```bash
terraform apply -var="aws_region=eu-west-1"
```

Or create a `terraform.tfvars` file:
```hcl
aws_region  = "eu-west-1"
bucket_name = "my-custom-bucket-name"
```

## Notes

- The S3 bucket name must be globally unique. If `spamail-bucket` is taken, update the variable.
- Lambda timeout is set to 300 seconds (5 minutes) to handle large batches.
- Versioning is enabled on the S3 bucket for data protection.
- CloudWatch logs retention is set to 7 days.
