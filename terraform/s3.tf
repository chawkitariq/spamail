resource "aws_s3_bucket" "spamail_bucket" {
  bucket        = "${local.prefix_name}-bucket"
  force_destroy = true
}

resource "aws_s3_object" "processed_folder" {
  bucket  = aws_s3_bucket.spamail_bucket.id
  key     = "processed/"
  content = ""
}

resource "aws_s3_object" "raw_ham_folder" {
  bucket  = aws_s3_bucket.spamail_bucket.id
  key     = "raw/ham/"
  content = ""
}

resource "aws_s3_object" "raw_spam_folder" {
  bucket  = aws_s3_bucket.spamail_bucket.id
  key     = "raw/spam/"
  content = ""
}

resource "aws_s3_object" "models_folder" {
  bucket  = aws_s3_bucket.spamail_bucket.id
  key     = "models/"
  content = ""
}

resource "aws_s3_bucket_server_side_encryption_configuration" "spamail_bucket" {
  bucket = aws_s3_bucket.spamail_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "spamail_bucket" {
  bucket = aws_s3_bucket.spamail_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_notification" "bucket_notification" {
  bucket = aws_s3_bucket.spamail_bucket.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.preprocess_lambda.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "raw/ham/"
    filter_suffix       = "_COMPLETE"
  }

  lambda_function {
    lambda_function_arn = aws_lambda_function.preprocess_lambda.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "raw/spam/"
    filter_suffix       = "_COMPLETE"
  }

  depends_on = [aws_lambda_permission.allow_s3_invoke]
}
