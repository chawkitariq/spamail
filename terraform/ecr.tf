resource "aws_ecr_repository" "spamail" {
  name                 = "spamail"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}
