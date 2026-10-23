variable "region" {
  type    = string
  default = "us-east-1"
}

variable "aws_profile" {
  type        = string
  description = "Local AWS profile (~/.aws/credentials) used by the provider."
  default     = "trustedlake"
}

variable "images_bucket" {
  type        = string
  description = "Bucket (existing) where the app uploads the ECG photos."
  default     = "ecg-lambdas-images-785236377332-us-east-1"
}

variable "results_bucket" {
  type        = string
  description = "Bucket (existing) where the reading writes the overlay and signal."
  default     = "ecg-lambdas-results-785236377332-us-east-1"
}

variable "code_version" {
  type        = string
  description = "Version exposed in meta.engine_version (e.g. git sha)."
  default     = "dev"
}

variable "mock_scenario" {
  type        = string
  description = "Default scenario when the payload has no mock.scenario. Empty = success."
  default     = ""

  validation {
    condition     = contains(["", "success", "success_with_warnings", "failure", "refused"], var.mock_scenario)
    error_message = "mock_scenario must be empty, success, success_with_warnings, failure or refused."
  }
}

variable "reading_image_tag" {
  type        = string
  description = "Tag of the real reading image in the ECR repo ecg-analyser."
  default     = "v6-quiet"
}
