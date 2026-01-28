resource "aws_ecr_repository" "spamail" {
  name                 = "spamail"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  tags = {
    Name        = "spamail"
    Description = "Stores all spamail images: preprocess-lambda, inference-lambda, etc."
  }
}
