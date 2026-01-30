data "aws_ecr_lifecycle_policy_document" "keep_last_10" {
  rule {
    priority    = 1
    description = "Keep last 10 images"

    selection {
      tag_status   = "any"
      count_type   = "imageCountMoreThan"
      count_number = 10
    }

    action {
      type = "expire"
    }
  }
}

resource "aws_ecr_repository" "spamail" {
  name                 = local.prefix_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

resource "aws_ecr_lifecycle_policy" "spamail" {
  repository = aws_ecr_repository.spamail.name
  policy     = data.aws_ecr_lifecycle_policy_document.keep_last_10.json
}