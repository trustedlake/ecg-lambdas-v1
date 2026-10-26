# CI/CD of the lambdas: CodePipeline V2 (QUEUED) <- GitHub trustedlake/ecg-lambdas-v1 (main) -> CodeBuild test || build -> deploy.
# Buildspecs in ../../ci. The deploy swaps the code (update-function-code), runs smoke_aws.py and rolls back if it fails.
# terraform apply does not touch the code (ignore_changes in main.tf).

variable "pipeline_repo" {
  type    = string
  default = "trustedlake/ecg-lambdas-v1"
}

variable "pipeline_branch" {
  type    = string
  default = "main"
}

variable "github_connection_name" {
  type        = string
  description = "CodeStar connection of the trustedlake GitHub account (owned by the ecg-backend stack)."
  default     = "ecg-github-frontend"
}

variable "models_bucket" {
  type        = string
  description = "Versioned bucket (created by hand) with model_selector.pkl, which is not in git."
  default     = "ecg-lambdas-models-785236377332-us-east-1"
}

data "aws_caller_identity" "current" {}

data "aws_codestarconnections_connection" "github" {
  name = var.github_connection_name
}

data "aws_sns_topic" "alerts" {
  name = "ecg-alerts"
}

data "aws_s3_bucket" "models" {
  bucket = var.models_bucket
}

locals {
  account_id = data.aws_caller_identity.current.account_id
  stages     = toset(["test", "build", "deploy"])
}

# --- Artifacts and logs ---

resource "aws_s3_bucket" "pipeline" {
  bucket = "ecg-lambdas-pipeline-${local.account_id}-${var.region}"
}

resource "aws_s3_bucket_public_access_block" "pipeline" {
  bucket                  = aws_s3_bucket.pipeline.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "pipeline" {
  bucket = aws_s3_bucket.pipeline.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "pipeline" {
  bucket = aws_s3_bucket.pipeline.id
  rule {
    id     = "expire-artifacts"
    status = "Enabled"
    filter {}
    expiration {
      days = 30
    }
  }
}

resource "aws_cloudwatch_log_group" "codebuild" {
  name              = "/codebuild/ecg-lambdas"
  retention_in_days = 30
}

# --- CodeBuild roles (one per stage) ---

data "aws_iam_policy_document" "codebuild_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["codebuild.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

locals {
  codebuild_base_statements = [
    {
      sid       = "Logs"
      actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
      resources = ["${aws_cloudwatch_log_group.codebuild.arn}:*"]
    },
    {
      sid       = "PipelineArtifacts"
      actions   = ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject"]
      resources = ["${aws_s3_bucket.pipeline.arn}/*"]
    },
  ]

  codebuild_stage_statements = {
    test = []
    build = [
      {
        sid       = "EcrLogin"
        actions   = ["ecr:GetAuthorizationToken"]
        resources = ["*"]
      },
      {
        sid = "EcrPush"
        actions = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:CompleteLayerUpload",
          "ecr:DescribeImages",
          "ecr:GetDownloadUrlForLayer",
          "ecr:InitiateLayerUpload",
          "ecr:PutImage",
          "ecr:UploadLayerPart",
        ]
        resources = [aws_ecr_repository.analyser.arn]
      },
      {
        sid       = "Model"
        actions   = ["s3:GetObject"]
        resources = ["${data.aws_s3_bucket.models.arn}/model_selector.pkl"]
      },
    ]
    deploy = [
      {
        sid = "Lambdas"
        actions = [
          "lambda:GetFunction",
          "lambda:GetFunctionConfiguration",
          "lambda:UpdateFunctionConfiguration",
          "lambda:UpdateFunctionCode",
          "lambda:InvokeFunction",
        ]
        resources = [aws_lambda_function.reading_real.arn, aws_lambda_function.this["interpretation"].arn]
      },
      {
        # Lambda checks that the caller can pull the image when it is set as the function code.
        sid       = "EcrLogin"
        actions   = ["ecr:GetAuthorizationToken"]
        resources = ["*"]
      },
      {
        sid       = "EcrPull"
        actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"]
        resources = [aws_ecr_repository.analyser.arn]
      },
      {
        # smoke_aws.py uploads a test image, reads the results of the reading and deletes the test image.
        sid       = "SmokeImages"
        actions   = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"]
        resources = ["${data.aws_s3_bucket.images.arn}/mock-smoke/*"]
      },
      {
        sid       = "SmokeResults"
        actions   = ["s3:GetObject"]
        resources = ["${data.aws_s3_bucket.results.arn}/*"]
      },
    ]
  }
}

resource "aws_iam_role" "codebuild" {
  for_each           = local.stages
  name               = "ecg-lambdas-codebuild-${each.key}-${var.region}"
  assume_role_policy = data.aws_iam_policy_document.codebuild_assume.json
}

data "aws_iam_policy_document" "codebuild" {
  for_each = local.stages

  dynamic "statement" {
    for_each = concat(local.codebuild_base_statements, local.codebuild_stage_statements[each.key])
    content {
      sid       = statement.value.sid
      actions   = statement.value.actions
      resources = statement.value.resources
    }
  }
}

resource "aws_iam_role_policy" "codebuild" {
  for_each = local.stages
  name     = "ecg-lambdas-codebuild-${each.key}"
  role     = aws_iam_role.codebuild[each.key].id
  policy   = data.aws_iam_policy_document.codebuild[each.key].json
}

# --- CodeBuild projects (x86_64: the reading image is x86_64) ---

locals {
  codebuild_env = {
    test = {}
    build = {
      ECR_REPOSITORY_URL = aws_ecr_repository.analyser.repository_url
      MODELS_BUCKET      = data.aws_s3_bucket.models.id
    }
    deploy = {
      ECR_REPOSITORY_URL      = aws_ecr_repository.analyser.repository_url
      FUNCTION_READING        = aws_lambda_function.reading_real.function_name
      FUNCTION_INTERPRETATION = aws_lambda_function.this["interpretation"].function_name
    }
  }
}

resource "aws_codebuild_project" "stage" {
  for_each = local.stages

  name          = "ecg-lambdas-${each.key}"
  service_role  = aws_iam_role.codebuild[each.key].arn
  build_timeout = each.key == "deploy" ? 40 : 30 # the real reading in the smoke test takes minutes

  source {
    type      = "CODEPIPELINE"
    buildspec = "ci/buildspec-${each.key}.yml"
  }

  artifacts {
    type = "CODEPIPELINE"
  }

  environment {
    type            = "LINUX_CONTAINER"
    compute_type    = each.key == "test" ? "BUILD_GENERAL1_MEDIUM" : "BUILD_GENERAL1_SMALL"
    image           = "aws/codebuild/standard:7.0"
    privileged_mode = each.key == "build"

    dynamic "environment_variable" {
      for_each = local.codebuild_env[each.key]
      content {
        name  = environment_variable.key
        value = environment_variable.value
      }
    }
  }

  logs_config {
    cloudwatch_logs {
      group_name  = aws_cloudwatch_log_group.codebuild.name
      stream_name = each.key
    }
  }
}

# --- Pipeline ---

data "aws_iam_policy_document" "codepipeline_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["codepipeline.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_iam_role" "codepipeline" {
  name               = "ecg-lambdas-codepipeline-${var.region}"
  assume_role_policy = data.aws_iam_policy_document.codepipeline_assume.json
}

data "aws_iam_policy_document" "codepipeline" {
  statement {
    sid       = "Artifacts"
    actions   = ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject", "s3:GetBucketVersioning"]
    resources = [aws_s3_bucket.pipeline.arn, "${aws_s3_bucket.pipeline.arn}/*"]
  }

  statement {
    sid       = "GitHub"
    actions   = ["codestar-connections:UseConnection", "codeconnections:UseConnection"]
    resources = [data.aws_codestarconnections_connection.github.arn]
  }

  statement {
    sid       = "Builds"
    actions   = ["codebuild:StartBuild", "codebuild:BatchGetBuilds"]
    resources = [for p in aws_codebuild_project.stage : p.arn]
  }
}

resource "aws_iam_role_policy" "codepipeline" {
  name   = "ecg-lambdas-codepipeline"
  role   = aws_iam_role.codepipeline.id
  policy = data.aws_iam_policy_document.codepipeline.json
}

resource "aws_codepipeline" "this" {
  name           = "ecg-lambdas"
  role_arn       = aws_iam_role.codepipeline.arn
  pipeline_type  = "V2"
  execution_mode = "QUEUED"

  artifact_store {
    type     = "S3"
    location = aws_s3_bucket.pipeline.bucket
  }

  stage {
    name = "Source"

    action {
      name             = "Source"
      category         = "Source"
      owner            = "AWS"
      provider         = "CodeStarSourceConnection"
      version          = "1"
      output_artifacts = ["source"]
      configuration = {
        ConnectionArn    = data.aws_codestarconnections_connection.github.arn
        FullRepositoryId = var.pipeline_repo
        BranchName       = var.pipeline_branch
      }
    }
  }

  # Test and Build run in parallel; Deploy needs both.
  stage {
    name = "TestAndBuild"

    action {
      name            = "Test"
      category        = "Build"
      owner           = "AWS"
      provider        = "CodeBuild"
      version         = "1"
      run_order       = 1
      input_artifacts = ["source"]
      configuration   = { ProjectName = aws_codebuild_project.stage["test"].name }
    }

    action {
      name             = "Build"
      category         = "Build"
      owner            = "AWS"
      provider         = "CodeBuild"
      version          = "1"
      run_order        = 1
      input_artifacts  = ["source"]
      output_artifacts = ["image"]
      configuration    = { ProjectName = aws_codebuild_project.stage["build"].name }
    }
  }

  stage {
    name = "Deploy"
    action {
      name            = "Deploy"
      category        = "Build"
      owner           = "AWS"
      provider        = "CodeBuild"
      version         = "1"
      input_artifacts = ["source", "image"]
      configuration = {
        ProjectName   = aws_codebuild_project.stage["deploy"].name
        PrimarySource = "source"
      }
    }
  }

  trigger {
    provider_type = "CodeStarSourceConnection"
    git_configuration {
      source_action_name = "Source"
      push {
        branches {
          includes = [var.pipeline_branch]
        }
      }
    }
  }
}

# --- Failure alert on the existing ecg-alerts topic ---
# The topic policy (ecg-backend stack, infra/aws/cicd.tf) must allow this rule: see its EventBridgePublish statement.

resource "aws_cloudwatch_event_rule" "pipeline_failed" {
  name        = "ecg-lambdas-pipeline-failed"
  description = "ecg-lambdas pipeline execution failed"
  event_pattern = jsonencode({
    source      = ["aws.codepipeline"]
    detail-type = ["CodePipeline Pipeline Execution State Change"]
    detail = {
      pipeline = [aws_codepipeline.this.name]
      state    = ["FAILED"]
    }
  })
}

resource "aws_cloudwatch_event_target" "pipeline_failed" {
  rule = aws_cloudwatch_event_rule.pipeline_failed.name
  arn  = data.aws_sns_topic.alerts.arn

  input_transformer {
    input_paths = {
      pipeline  = "$.detail.pipeline"
      execution = "$.detail.execution-id"
    }
    input_template = "\"Pipeline <pipeline> failed (execution <execution>). See CodePipeline in the console, region ${var.region}.\""
  }
}

output "pipeline_name" {
  value = aws_codepipeline.this.name
}
