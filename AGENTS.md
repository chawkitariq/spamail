# Agent Instructions for Spamail

This repository is an MLOps project that uses AWS, Terraform, DVC, and SageMaker for preprocessing, training, and deployment.

## Start Here
- Project overview and local workflow: [README.md](README.md)
- AWS infrastructure and deployment flow: [terraform/README.md](terraform/README.md)

## Key Directories
- src/ contains local scripts for preprocessing, training, and inference.
- sagemaker/ contains Dockerized preprocessing, training, and inference apps used by SageMaker.
- terraform/ defines AWS infrastructure, SageMaker pipeline, API Gateway, and ECR build/push.
- datas/ and models/ hold DVC-tracked artifacts (see *.dvc files).

## Common Commands
Local (Python):
- python3 src/preprocess.py
- python3 src/train.py
- python3 src/inference.py

Infra (Terraform):
- cd terraform
- terraform init -backend-config="bucket=..." -backend-config="key=..." -backend-config="region=..."
- terraform plan
- terraform apply

AWS (Pipeline trigger):
- aws s3 rsync ../datas/raw/ham/ s3://<bucket>/raw/ham/ --recursive
- aws s3 rsync ../datas/raw/spam/ s3://<bucket>/raw/spam/ --recursive
- aws sagemaker start-pipeline-execution --pipeline-name <prefix>-pipeline --region <region>

## DVC Notes
- Large artifacts are tracked via .dvc files under datas/ and models/.
- Use dvc pull to fetch data/model artifacts and dvc push to publish updates.

## Conventions and Pitfalls
- SageMaker images are built from sagemaker/* and pushed via Terraform (see docker_build_push.tf).
- The SageMaker pipeline reads raw data from S3 and writes processed CSV and model artifacts back to S3.
- Keep AWS region, bucket name, and pipeline name consistent with terraform variables and outputs.
