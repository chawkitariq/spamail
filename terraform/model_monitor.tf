data "aws_sagemaker_prebuilt_ecr_image" "model_monitor" {
  repository_name = "sagemaker-model-monitor-analyzer"
  region          = var.aws_region
}

resource "aws_sagemaker_data_quality_job_definition" "spamail" {
  name     = "${local.prefix_name}-data-quality-job"
  role_arn = aws_iam_role.sagemaker_execution.arn

  data_quality_app_specification {
    image_uri = data.aws_sagemaker_prebuilt_ecr_image.model_monitor.registry_path
  }

  data_quality_baseline_config {
    constraints_resource {
      s3_uri = "s3://${aws_s3_bucket.spamail_bucket.id}/monitoring/baseline/constraints.json"
    }
    statistics_resource {
      s3_uri = "s3://${aws_s3_bucket.spamail_bucket.id}/monitoring/baseline/statistics.json"
    }
  }

  data_quality_job_input {
    endpoint_input {
      endpoint_name             = local.sagemaker_endpoint_name
      local_path                = "/opt/ml/processing/input/endpoint"
      s3_data_distribution_type = "FullyReplicated"
      s3_input_mode             = "File"
    }
  }

  data_quality_job_output_config {
    monitoring_outputs {
      s3_output {
        s3_uri         = "s3://${aws_s3_bucket.spamail_bucket.id}/monitoring/reports/"
        local_path     = "/opt/ml/processing/output"
        s3_upload_mode = "EndOfJob"
      }
    }
  }

  job_resources {
    cluster_config {
      instance_count    = 1
      instance_type     = "ml.m5.large"
      volume_size_in_gb = 20
    }
  }

  network_config {
    enable_inter_container_traffic_encryption = false
    enable_network_isolation                  = false
  }

  stopping_condition {
    max_runtime_in_seconds = 3600
  }

  depends_on = [
    aws_sagemaker_pipeline.spamail
  ]
}

resource "aws_sagemaker_monitoring_schedule" "spamail_data_quality" {
  name = "${local.prefix_name}-data-quality-monitor"

  monitoring_schedule_config {
    monitoring_job_definition_name = aws_sagemaker_data_quality_job_definition.spamail.name
    monitoring_type                = "DataQuality"

    schedule_config {
      schedule_expression = "cron(0 * * * ? *)" # Hourly
    }
  }

  lifecycle {
    ignore_changes = [monitoring_schedule_config[0].schedule_config[0].schedule_expression]
  }

  depends_on = [
    aws_sagemaker_data_quality_job_definition.spamail
  ]
}
