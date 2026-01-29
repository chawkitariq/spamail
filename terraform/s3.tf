resource "aws_s3_bucket" "spamail_bucket" {
  bucket        = "spamail-bucket"
  force_destroy = true
}

resource "aws_s3_bucket_notification" "bucket_notification" {
  bucket = aws_s3_bucket.spamail_bucket.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.preprocess_lambda.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "raw/ham/_COMPLETE"
  }

  lambda_function {
    lambda_function_arn = aws_lambda_function.preprocess_lambda.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "raw/spam/_COMPLETE"
  }

  depends_on = [aws_lambda_permission.allow_s3_invoke]
}

resource "aws_s3_object" "raw_ham_folder" {
  bucket  = aws_s3_bucket.spamail_bucket.id
  key     = "raw/ham/.gitkeep"
  content = ""
}

resource "aws_s3_object" "raw_spam_folder" {
  bucket  = aws_s3_bucket.spamail_bucket.id
  key     = "raw/spam/.gitkeep"
  content = ""
}

resource "aws_s3_object" "processed_folder" {
  bucket  = aws_s3_bucket.spamail_bucket.id
  key     = "processed/.gitkeep"
  content = ""
}
