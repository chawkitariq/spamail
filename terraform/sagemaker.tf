resource "aws_sagemaker_pipeline" "spamail" {
  pipeline_name         = "${local.prefix_name}-pipeline"
  pipeline_display_name = "${var.project_name}-Pipeline"
  pipeline_description  = "ML pipeline for training and deploying ${var.project_name} spam classifier"
  role_arn              = aws_iam_role.sagemaker_execution.arn

  pipeline_definition = jsonencode({
    Version = "2020-12-01"

    Steps = [
      {
        Name = "PreprocessEmails"
        DisplayName = "Preprocess Raw Emails to CSV"
        Type = "Processing"
        Arguments = {
          ProcessingResources = {
            ClusterConfig = {
              InstanceCount  = 1
              InstanceType   = "ml.m5.large"
              VolumeSizeInGB = 10
            }
          }
          AppSpecification = {
            ImageUri = "${aws_ecr_repository.spamail.repository_url}:preprocess-latest"
          }
          ProcessingInputs = [
            {
              InputName = "raw"
              S3Input = {
                S3Uri                  = "s3://${aws_s3_bucket.spamail_bucket.id}/raw/"
                LocalPath              = "/opt/ml/processing/input"
                S3DataType             = "S3Prefix"
                S3InputMode            = "File"
                S3DataDistributionType = "FullyReplicated"
              }
            }
          ]
          ProcessingOutputConfig = {
            Outputs = [
              {
                OutputName = "preprocessed-data"
                S3Output = {
                  S3Uri        = "s3://${aws_s3_bucket.spamail_bucket.id}/processed/"
                  LocalPath    = "/opt/ml/processing/output"
                  S3UploadMode = "EndOfJob"
                }
              }
            ]
          }
          RoleArn = aws_iam_role.sagemaker_execution.arn
        }
      },

      {
        Name      = "TrainModel"
        DisplayName = "Train Spam Classifier Model"
        Type      = "Training"
        DependsOn = ["PreprocessEmails"]
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
            InstanceType   = "ml.m5.large"
            VolumeSizeInGB = 5
          }
          RoleArn = aws_iam_role.sagemaker_execution.arn
          StoppingCondition = {
            MaxRuntimeInSeconds = 3600
          }
        }
      },

      {
        Name      = "CreateModel"
        DisplayName = "Create SageMaker Model Artifact"
        Type      = "Model"
        DependsOn = ["TrainModel"]
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
        Name : "DeployModel_EndpointConfig"
        DisplayName : "Create Endpoint Config for Deployment"
        Type : "EndpointConfig"
        DependsOn : ["CreateModel"]
        Arguments : {
          ProductionVariants : [
            {
              VariantName : "AllTraffic"
              ServerlessConfig : {
                MemorySizeInMB : 2048
                MaxConcurrency : 4
              }
              ModelName : {
                Get : "Steps.CreateModel.ModelName"
              }
            }
          ]
        },
      },

      {
        Name : "DeployModel"
        DisplayName : "Deploy Model as Serverless Endpoint"
        Type : "Endpoint"
        DependsOn : ["DeployModel_EndpointConfig"]
        Arguments : {
          EndpointName : local.sagemaker_endpoint_name
          EndpointConfigName : {
            Get : "Steps.DeployModel_EndpointConfig.EndpointConfigName"
          }
        }
      }
    ]
  })
}
