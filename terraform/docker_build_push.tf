resource "null_resource" "docker_build_push_preprocess" {
  provisioner "local-exec" {
    command = <<EOT
      docker build -t ${aws_ecr_repository.spamail.repository_url}:preprocess-lambda-latest -f ${path.module}/../lambdas/Dockerfile ${path.module}/../lambdas/preprocess/ --network host
      aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.spamail.repository_url}
      docker push ${aws_ecr_repository.spamail.repository_url}:preprocess-lambda-latest
    EOT
  }

  triggers = {
    dockerfile_hash = filemd5("${path.module}/../lambdas/Dockerfile")
    all_files_hash = md5(join("", [for f in fileset("${path.module}/../lambdas/preprocess", "**") : filemd5("${path.module}/../lambdas/preprocess/${f}")]))
  }

  depends_on = [aws_ecr_repository.spamail]
}

resource "null_resource" "docker_build_push_train" {
  provisioner "local-exec" {
    command = <<EOT
      docker build -t ${aws_ecr_repository.spamail.repository_url}:train-latest -f ${path.module}/../sagemaker/train/Dockerfile ${path.module}/../sagemaker/train/ --network host
      aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.spamail.repository_url}
      docker push ${aws_ecr_repository.spamail.repository_url}:train-latest
    EOT
  }
  triggers = {
    all_files_hash = md5(join("", [for f in fileset("${path.module}/../sagemaker/train", "**") : filemd5("${path.module}/../sagemaker/train/${f}")]))
  }

  depends_on = [aws_ecr_repository.spamail]
}

resource "null_resource" "docker_build_push_inference" {
  provisioner "local-exec" {
    command = <<EOT
      docker build -t ${aws_ecr_repository.spamail.repository_url}:inference-latest -f ${path.module}/../sagemaker/inference/Dockerfile ${path.module}/../sagemaker/inference/ --network host
      aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.spamail.repository_url}
      docker push ${aws_ecr_repository.spamail.repository_url}:inference-latest
    EOT
  }

  triggers = {
    all_files_hash = md5(join("", [for f in fileset("${path.module}/../sagemaker/inference", "**") : filemd5("${path.module}/../sagemaker/inference/${f}")]))
  }

  depends_on = [aws_ecr_repository.spamail]
}

