resource "aws_sns_topic" "model_monitor_alerts" {
  name = "${local.prefix_name}-model-monitor-alerts"
}

resource "aws_sns_topic_subscription" "model_monitor_email" {
  topic_arn = aws_sns_topic.model_monitor_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}
