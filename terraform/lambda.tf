resource "aws_lambda_function" "preprocess_lambda" {
  function_name = "${local.prefix_name}-preprocess"
  role          = aws_iam_role.lambda_role.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.spamail.repository_url}:preprocess-lambda-latest"
  timeout       = 300
  memory_size   = 1024

  environment {
    variables = {
      BUCKET_NAME = aws_s3_bucket.spamail_bucket.id
    }
  }

  depends_on = [
    aws_ecr_repository.spamail,
    null_resource.docker_build_push_preprocess
  ]
}

resource "aws_lambda_permission" "allow_s3_invoke" {
  statement_id  = "AllowExecutionFromS3"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.preprocess_lambda.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.spamail_bucket.arn
}
