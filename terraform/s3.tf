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

resource "aws_s3_object" "monitoring_datacapture_folder" {
  bucket  = aws_s3_bucket.spamail_bucket.id
  key     = "monitoring/datacapture/"
  content = ""
}

resource "aws_s3_object" "monitoring_baseline_folder" {
  bucket  = aws_s3_bucket.spamail_bucket.id
  key     = "monitoring/baseline/"
  content = ""
}

resource "aws_s3_object" "monitoring_reports_folder" {
  bucket  = aws_s3_bucket.spamail_bucket.id
  key     = "monitoring/reports/"
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

