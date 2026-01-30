# SageMaker Model Package Group
resource "aws_sagemaker_model_package_group" "spamail" {
  model_package_group_name        = "${local.prefix_name}-model-group"
  model_package_group_description = "Model package group for ${var.project_name} spam classifier"

  provisioner "local-exec" {
    when    = destroy
    command = "aws sagemaker delete-model-package-group --model-package-group-name ${self.model_package_group_name} || true"
  }
}

# SageMaker Pipeline
resource "aws_sagemaker_pipeline" "spamail" {
  pipeline_name         = "${local.prefix_name}-pipeline"
  pipeline_display_name = "${var.project_name}-Pipeline"
  pipeline_description  = "ML pipeline for training and deploying ${var.project_name} spam classifier"
  role_arn              = aws_iam_role.sagemaker_execution.arn

  pipeline_definition = jsonencode({
    Version = "2020-12-01"

    Steps = [
      {
        Name = "TrainModel"
        Type = "Training"
        Arguments = {
          AlgorithmSpecification = {
            TrainingImage     = "${aws_ecr_repository.spamail.repository_url}:train-latest"
            TrainingInputMode = "File"
          }
          InputDataConfig = [
            {
              ChannelName = "train"
              DataSource = {
                S3DataSource = {
                  S3DataType             = "S3Prefix"
                  S3Uri                  = "s3://${aws_s3_bucket.spamail_bucket.id}/processed/"
                  S3DataDistributionType = "FullyReplicated"
                }
              }
              ContentType = "text/csv"
            }
          ]
          OutputDataConfig = {
            S3OutputPath = "s3://${aws_s3_bucket.spamail_bucket.id}/models/"
          }
          ResourceConfig = {
            InstanceCount  = 1
            InstanceType   = var.training_instance_type
            VolumeSizeInGB = 5
          }
          RoleArn = aws_iam_role.sagemaker_execution.arn
          StoppingCondition = {
            MaxRuntimeInSeconds = 3600
          }
        }
      },

      {
        Name      = "RegisterModel"
        Type      = "RegisterModel"
        DependsOn = ["TrainModel"]
        Arguments = {
          ModelPackageGroupName = aws_sagemaker_model_package_group.spamail.model_package_group_name
          ModelApprovalStatus   = "PendingManualApproval"
          InferenceSpecification = {
            Containers = [
              {
                Image = "${aws_ecr_repository.spamail.repository_url}:inference-latest"
                ModelDataUrl = {
                  "Get" = "Steps.TrainModel.ModelArtifacts.S3ModelArtifacts"
                }
              }
            ]
            SupportedContentTypes      = ["application/json"]
            SupportedResponseMIMETypes = ["application/json"]
          }
        }
      },

      {
        Name      = "CreateModel"
        Type      = "Model"
        DependsOn = ["RegisterModel"]
        Arguments = {
          ExecutionRoleArn = aws_iam_role.sagemaker_execution.arn
          PrimaryContainer = {
            Image = "${aws_ecr_repository.spamail.repository_url}:inference-latest"
            ModelDataUrl = {
              "Get" = "Steps.TrainModel.ModelArtifacts.S3ModelArtifacts"
            }
          }
        }
      },

      {
        Name : "DeployModel_EndpointConfig",
        Type : "EndpointConfig",
        DependsOn : ["CreateModel"],
        Arguments : {
          ProductionVariants : [
            {
              VariantName : "AllTraffic",
              ServerlessConfig : {
                MemorySizeInMB : 2048,
                MaxConcurrency : 4
              },
              ModelName : {
                Get : "Steps.CreateModel.ModelName"
              },
            }
          ]
        },
      },

      {
        Name : "DeployModel",
        Type : "Endpoint",
        DependsOn : ["DeployModel_EndpointConfig"],
        DisplayName : "Deploy model (endpoint)",
        Arguments : {
          EndpointName : local.sagemaker_endpoint_name,
          EndpointConfigName : {
            Get : "Steps.DeployModel_EndpointConfig.EndpointConfigName"
          }
        }
      }
    ]
  })

  depends_on = [
    aws_sagemaker_model_package_group.spamail,
  ]
}
