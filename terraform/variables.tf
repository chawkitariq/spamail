variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "eu-west-3"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "spamail"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "domain_name" {
  description = "Custom domain name for API Gateway (e.g., api.example.com). Leave empty to skip custom domain setup."
  type        = string
  default     = "spamail.chawkitariq.fr"
}

variable "route53_zone_id" {
  description = "Route 53 hosted zone ID for the domain. Required if domain_name is set."
  type        = string
}

variable "alert_email" {
  description = "Email address for Model Monitor alerts. Leave empty to skip email notifications."
  type        = string
}
