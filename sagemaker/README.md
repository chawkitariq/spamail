# SageMaker ML Deployment

This document describes the SageMaker implementation for training and deploying the spam classifier.

## Architecture

- **SageMaker Pipeline**: Automated ML workflow (train → register → create model → deploy)
- **ECR Repositories**: Separate repos for training and inference Docker images
- **S3 Bucket**: Stores training data and model artifacts
- **API Gateway**: REST API endpoint for predictions
- **Lambda**: Deploys SageMaker endpoints
- **CloudWatch**: Monitoring and logs

## Quick Start

### 1. Build and Push Docker Images

```bash
# Build training image
cd sagemaker/train
docker build -t <account-id>.dkr.ecr.eu-west-3.amazonaws.com/dev-spamail-training:latest .

# Push training image
aws ecr get-login-password --region eu-west-3 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-west-3.amazonaws.com
docker push <account-id>.dkr.ecr.eu-west-3.amazonaws.com/dev-spamail-training:latest

# Build inference image
cd ../inference
docker build -t <account-id>.dkr.ecr.eu-west-3.amazonaws.com/dev-spamail-inference:latest .
docker push <account-id>.dkr.ecr.eu-west-3.amazonaws.com/dev-spamail-inference:latest
```

### 2. Upload Training Data

```bash
# Upload the processed email.csv to S3
aws s3 cp datas/processed/email.csv s3://dev-spamail-ml/input/
```

### 3. Start SageMaker Pipeline

```bash
# Execute the pipeline via AWS CLI
aws sagemaker start-pipeline-execution \
  --pipeline-name dev-spamail-pipeline \
  --region eu-west-3
```

Or use the AWS Console to start the pipeline.

### 4. Test the API

Once the pipeline completes and the endpoint is deployed:

```bash
curl -X POST https://<api-gateway-url>/prod/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"text": "Congratulations! You have won a free prize!"},
      {"text": "Hi team, please find the meeting notes attached."}
    ]
  }'
```

Expected response:
```json
{
  "predictions": [1, 0],
  "labels": ["spam", "ham"],
  "probabilities": [[0.05, 0.95], [0.98, 0.02]]
}
```

## Pipeline Steps

1. **TrainModel**: Trains the spam classifier on uploaded data
2. **RegisterModel**: Registers the model in the model registry
3. **CreateModel**: Creates a SageMaker model
4. **DeployEndpoint**: Deploys a serverless endpoint via Lambda

## Environment Variables

- `environment`: dev, staging, or prod
- `training_instance_type`: ml.m5.large (default)

## Monitoring

View pipeline execution logs in CloudWatch:
- `/aws/sagemaker/dev-spamail-pipeline`
- `/aws/apigateway/dev-spamail`

## Costs

- **Serverless Endpoint**: Pay per inference request
- **Training**: Pay per training hour (ml.m5.large)
- **Storage**: S3 storage for data and models
