# Lambda Function for Endpoint Deployment
data "archive_file" "preprocess_lambda" {
  type        = "zip"
  source_dir  = "${path.module}/../lambdas/preprocess"
  output_path = "${path.module}/../lambdas/preprocess/lambda_handler.zip"
}

# Lambda Function
resource "aws_lambda_function" "preprocess_lambda" {
  filename         = data.archive_file.preprocess_lambda.output_path
  function_name    = "spamail-preprocess"
  role            = aws_iam_role.lambda_role.arn
  handler         = "lambda_handler.lambda_handler"
  source_code_hash = filebase64sha256(data.archive_file.preprocess_lambda.output_path)
  runtime         = "python3.11"
  timeout         = 300
  memory_size     = 512

  environment {
    variables = {
      BUCKET_NAME = aws_s3_bucket.spamail_bucket.id
    }
  }
}

# Permission for S3 to invoke Lambda
resource "aws_lambda_permission" "allow_s3_invoke" {
  statement_id  = "AllowExecutionFromS3"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.preprocess_lambda.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.spamail_bucket.arn
}

