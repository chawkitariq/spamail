variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "eu-west-3"
}

variable "bucket_name" {
  description = "Name of the S3 bucket"
  type        = string
  default     = "spamail-bucket"
}
