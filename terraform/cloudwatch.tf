# CloudWatch Log Group for API Gateway
resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/${local.prefix_name}"
  retention_in_days = 7
}

# CloudWatch Log Group for SageMaker Pipeline
resource "aws_cloudwatch_log_group" "sagemaker_pipeline" {
  name              = "/aws/sagemaker/${local.prefix_name}-pipeline"
  retention_in_days = 7
}
