variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "eu-west-3"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "spamail"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "training_instance_type" {
  description = "EC2 instance type for SageMaker training jobs"
  type        = string
  default     = "ml.m5.large"
}
