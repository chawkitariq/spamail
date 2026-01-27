# IAM Role for Lambda
resource "aws_iam_role" "lambda_role" {
  name = "spamail_lambda_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# IAM Policy for Lambda to access S3 and CloudWatch Logs
resource "aws_iam_role_policy" "lambda_policy" {
  name = "spamail_lambda_policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${aws_s3_bucket.spamail_bucket.arn}",
          "${aws_s3_bucket.spamail_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject"
        ]
        Resource = [
          "${aws_s3_bucket.spamail_bucket.arn}/processed/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# Lambda Layer for pandas (since it's too large for Lambda)
resource "aws_lambda_layer_version" "pandas_layer" {
  filename            = "${path.module}/../lambdas/layers/pandas_layer.zip"
  layer_name          = "pandas-layer"
  compatible_runtimes = ["python3.11"]
  
  # You need to create this layer separately
  # See: https://github.com/keithrozario/Klayers for pre-built layers
}

# Lambda Function
resource "aws_lambda_function" "preprocess_lambda" {
  filename         = "${path.module}/../lambdas/preprocess/lambda_package.zip"
  function_name    = "spamail-preprocess"
  role            = aws_iam_role.lambda_role.arn
  handler         = "lambda_function.lambda_handler"
  source_code_hash = filebase64sha256("${path.module}/../lambdas/preprocess/lambda_package.zip")
  runtime         = "python3.11"
  timeout         = 300
  memory_size     = 512

  layers = [aws_lambda_layer_version.pandas_layer.arn]

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

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "lambda_log_group" {
  name              = "/aws/lambda/${aws_lambda_function.preprocess_lambda.function_name}"
  retention_in_days = 7
}
