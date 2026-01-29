locals {
  prefix_name             = "${var.environment}-${var.project_name}"
  sagemaker_endpoint_name = "${local.prefix_name}-endpoint"

  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}
