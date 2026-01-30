# SageMaker ML Deployment

This document describes the SageMaker implementation for training and deploying the spam classifier.

## Architecture

- **SageMaker Pipeline**: Automated ML workflow (preprocess → train → register → create model → deploy)
- **SageMaker Processing**: Batch email preprocessing step
- **ECR Repositories**: Separate repos for preprocessing, training and inference Docker images
- **S3 Bucket**: Stores training data and model artifacts
- **API Gateway**: REST API endpoint for predictions with custom domain
- **Route 53**: Custom domain configuration
- **CloudWatch**: Monitoring and logs

## Pipeline Steps

1. **PreprocessEmails**: Processes raw emails into structured CSV format
2. **TrainModel**: Trains the spam classifier on processed data
3. **RegisterModel**: Registers the model in the SageMaker Model Registry
4. **CreateModel**: Creates a SageMaker model from the registered version
5. **DeployEndpoint**: Deploys a serverless endpoint

## Custom Domain

The API is accessible via a custom domain configured with Route 53:

- **Domain**: api.spamail.dev (configurable per environment)
- **SSL**: Automatic certificate management via AWS Certificate Manager
- **Routing**: Route 53 handles DNS resolution to API Gateway

## Monitoring

View pipeline execution logs in CloudWatch:
- `/aws/sagemaker/dev-spamail-pipeline`
- `/aws/apigateway/dev-spamail`
