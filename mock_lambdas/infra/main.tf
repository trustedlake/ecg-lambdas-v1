# MOCK lambdas for ECG reading and interpretation (contract v1, LAMBDA_CONTRACT.md).
# Independent stack: own state, only references the existing ecg-lambdas
# buckets (without changing them). Run ../build.sh before plan/apply.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
  }

  backend "s3" {
    bucket = "ecg-lambdas-terraform-state"
    key    = "ecg-mock-lambdas/terraform.tfstate"
    region = "sa-east-1"
  }
}

provider "aws" {
  region  = "sa-east-1"
  profile = var.aws_profile

  default_tags {
    tags = {
      project = "ecg"
      stack   = "ecg-mock-lambdas"
    }
  }
}

data "aws_s3_bucket" "images" {
  bucket = var.images_bucket
}

data "aws_s3_bucket" "results" {
  bucket = var.results_bucket
}

locals {
  mock_env = var.mock_scenario == "" ? {} : { MOCK_SCENARIO = var.mock_scenario }
  functions = {
    interpretation = {
      name        = "ecg-interpretation"
      memory_size = 256
      timeout     = 30
      environment = merge({ CODE_VERSION = var.code_version }, local.mock_env)
    }
  }
}

data "archive_file" "package" {
  for_each    = local.functions
  type        = "zip"
  source_dir  = "${path.module}/../build/${each.key}"
  output_path = "${path.module}/../build/${each.key}.zip"
}

resource "aws_iam_role" "this" {
  for_each = local.functions
  name     = "${each.value.name}-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "basic" {
  for_each   = local.functions
  role       = aws_iam_role.this[each.key].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_cloudwatch_log_group" "this" {
  for_each          = local.functions
  name              = "/aws/lambda/${each.value.name}"
  retention_in_days = 30
}

resource "aws_lambda_function" "this" {
  for_each         = local.functions
  function_name    = each.value.name
  description      = "${each.key} ECG lambda — contract v1 (ecg-lambdas-v1/LAMBDA_CONTRACT.md)"
  role             = aws_iam_role.this[each.key].arn
  handler          = "handler.handler"
  runtime          = "python3.12"
  architectures    = ["arm64"]
  timeout          = each.value.timeout
  memory_size      = each.value.memory_size
  filename         = data.archive_file.package[each.key].output_path
  source_code_hash = data.archive_file.package[each.key].output_base64sha256

  environment {
    variables = each.value.environment
  }

  depends_on = [aws_cloudwatch_log_group.this, aws_iam_role_policy_attachment.basic]
}

# ---------------------------------------------------------------------------
# REAL reading (ecg-analyser, contract 2.0): Lambda as a container image.
# Image in the existing ECR repo "ecg-analyser", tag var.reading_image_tag.
# ---------------------------------------------------------------------------
data "aws_ecr_repository" "analyser" {
  name = "ecg-analyser"
}

resource "aws_iam_role" "reading_real" {
  name = "ecg-reading-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "reading_real_basic" {
  role       = aws_iam_role.reading_real.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "reading_real_s3" {
  name = "ecg-reading-s3"
  role = aws_iam_role.reading_real.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = "s3:GetObject", Resource = "${data.aws_s3_bucket.images.arn}/*" },
      { Effect = "Allow", Action = "s3:PutObject", Resource = "${data.aws_s3_bucket.results.arn}/reading/*" },
    ]
  })
}

resource "aws_cloudwatch_log_group" "reading_real" {
  name              = "/aws/lambda/ecg-reading"
  retention_in_days = 30
}

resource "aws_lambda_function" "reading_real" {
  function_name = "ecg-reading"
  description   = "REAL ECG reading (ecg-analyser, 6 methods, contract 2.0)"
  role          = aws_iam_role.reading_real.arn
  package_type  = "Image"
  image_uri     = "${data.aws_ecr_repository.analyser.repository_url}:${var.reading_image_tag}"
  architectures = ["x86_64"]
  timeout       = 900
  memory_size   = 3538 # 2 vCPU: the 6 methods run in subprocesses
  environment {
    variables = { RESULTS_BUCKET = data.aws_s3_bucket.results.id, CODE_VERSION = var.code_version }
  }
  depends_on = [aws_cloudwatch_log_group.reading_real, aws_iam_role_policy_attachment.reading_real_basic]
}
