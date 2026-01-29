resource "aws_ecr_repository" "spamail" {
  name                 = local.prefix_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}
