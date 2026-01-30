output "sagemaker_pipeline_name" {
  description = "Name of the SageMaker Pipeline"
  value       = aws_sagemaker_pipeline.spamail.pipeline_name
}

output "sagemaker_endpoint_name" {
  description = "Name of the SageMaker Endpoint"
  value       = local.sagemaker_endpoint_name
}

output "api_gateway_url" {
  description = "URL of the API Gateway endpoint"
  value       = "${aws_api_gateway_stage.default.invoke_url}/predict"
}

output "custom_domain_url" {
  description = "Custom domain URL (if configured)"
  value       = "https://${var.domain_name}/predict"
}
