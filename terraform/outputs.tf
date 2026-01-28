output "bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.spamail_bucket.id
}

output "bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.spamail_bucket.arn
}

output "lambda_function_name" {
  description = "Name of the Lambda function"
  value       = aws_lambda_function.preprocess_lambda.function_name
}

output "lambda_function_arn" {
  description = "ARN of the Lambda function"
  value       = aws_lambda_function.preprocess_lambda.arn
}

output "ecr_repository_url" {
  description = "URL of the ECR repository for all spamail images"
  value       = aws_ecr_repository.spamail.repository_url
}

output "ecr_repository_name" {
  description = "Name of the ECR repository"
  value       = aws_ecr_repository.spamail.name
}
