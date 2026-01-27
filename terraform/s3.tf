resource "aws_s3_bucket" "spamail_bucket" {
  bucket = "spamail-bucket"

  tags = {
    Name        = "Spamail Bucket"
    Environment = "production"
  }
}

resource "aws_s3_bucket_versioning" "spamail_bucket_versioning" {
  bucket = aws_s3_bucket.spamail_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_notification" "bucket_notification" {
  bucket = aws_s3_bucket.spamail_bucket.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.preprocess_lambda.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "raw/"
  }

  depends_on = [aws_lambda_permission.allow_s3_invoke]
}

# Create folder structure by uploading .gitkeep files
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
