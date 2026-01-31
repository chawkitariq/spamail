resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/${local.prefix_name}"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "sagemaker_pipeline" {
  name              = "/aws/sagemaker/${local.prefix_name}-pipeline"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "model_monitor" {
  name              = "/aws/sagemaker/ModelMonitor/${local.prefix_name}"
  retention_in_days = 7
}

resource "aws_cloudwatch_metric_alarm" "data_quality_violations" {
  alarm_name          = "${local.prefix_name}-data-quality-violations"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "feature_baseline_drift_total_minus_7_days"
  namespace           = "aws/sagemaker/Endpoints/data-metrics"
  period              = 3600
  statistic           = "Average"
  threshold           = 0.1
  alarm_description   = "This metric monitors data quality drift in the spam classifier endpoint"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.model_monitor_alerts.arn]

  dimensions = {
    Endpoint = local.sagemaker_endpoint_name
  }
}

resource "aws_cloudwatch_metric_alarm" "missing_data" {
  alarm_name          = "${local.prefix_name}-missing-data"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "feature_baseline_drift_text"
  namespace           = "aws/sagemaker/Endpoints/data-metrics"
  period              = 3600
  statistic           = "Average"
  threshold           = 0.2
  alarm_description   = "This metric monitors for missing or corrupted input data"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.model_monitor_alerts.arn]

  dimensions = {
    Endpoint = local.sagemaker_endpoint_name
  }
}
