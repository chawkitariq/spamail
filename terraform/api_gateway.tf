# API Gateway REST API
resource "aws_api_gateway_rest_api" "main" {
  name        = "${local.prefix_name}-api"
  description = "REST API for ${local.prefix_name} spam classifier predictions"
}

# Root resource ("/predict")
resource "aws_api_gateway_resource" "predict" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  parent_id   = aws_api_gateway_rest_api.main.root_resource_id
  path_part   = "predict"
}

# Method for POST /predict
resource "aws_api_gateway_method" "predict_post" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  resource_id   = aws_api_gateway_resource.predict.id
  http_method   = "POST"
  authorization = "NONE"
}

# Integration with SageMaker Runtime
resource "aws_api_gateway_integration" "sagemaker" {
  rest_api_id             = aws_api_gateway_rest_api.main.id
  resource_id             = aws_api_gateway_resource.predict.id
  http_method             = aws_api_gateway_method.predict_post.http_method
  integration_http_method = "POST"
  type                    = "AWS"
  uri                     = "arn:aws:apigateway:${data.aws_region.current.name}:runtime.sagemaker:path//endpoints/${local.sagemaker_endpoint_name}/invocations"
  credentials             = aws_iam_role.apigateway_sagemaker_role.arn

  request_templates = {
    "application/json" = "$input.body"
  }
}

# Method response for 200
resource "aws_api_gateway_method_response" "response_200" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.predict.id
  http_method = aws_api_gateway_method.predict_post.http_method
  status_code = "200"

  response_models = {
    "application/json" = "Empty"
  }
}

# Integration response
resource "aws_api_gateway_integration_response" "response" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  resource_id = aws_api_gateway_resource.predict.id
  http_method = aws_api_gateway_method.predict_post.http_method
  status_code = aws_api_gateway_method_response.response_200.status_code

  response_templates = {
    "application/json" = "$input.body"
  }

  depends_on = [aws_api_gateway_integration.sagemaker]
}

# Deployment
resource "aws_api_gateway_deployment" "main" {
  rest_api_id = aws_api_gateway_rest_api.main.id
  
  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.predict.id,
      aws_api_gateway_method.predict_post.id,
      aws_api_gateway_integration.sagemaker.id,
      aws_api_gateway_integration_response.response.id,
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }

  depends_on = [
    aws_api_gateway_integration.sagemaker,
    aws_api_gateway_integration_response.response
  ]
}

resource "aws_api_gateway_account" "cloudwatch_role" {
  cloudwatch_role_arn = aws_iam_role.apigateway_sagemaker_role.arn
}

# Stage
resource "aws_api_gateway_stage" "default" {
  rest_api_id   = aws_api_gateway_rest_api.main.id
  deployment_id = aws_api_gateway_deployment.main.id
  stage_name    = "prod"

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      resourcePath   = "$context.resourcePath"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }

  depends_on = [
    aws_api_gateway_account.cloudwatch_role
  ]
}
