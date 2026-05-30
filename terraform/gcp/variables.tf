variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "europe-central2"
}

variable "service_name" {
  description = "Name of the Cloud Run service"
  type        = string
  default     = "doc-research-agent"
}

variable "docker_image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}

variable "git_sha" {
  description = "Git commit SHA, stamped into image for log forensics"
  type        = string
  default     = "unknown"
}

variable "app_version" {
  description = "Human-readable app version (e.g. git describe output)"
  type        = string
  default     = "unknown"
}

variable "openai_api_key" {
  description = "OpenAI API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "openrouter_api_key" {
  description = "OpenRouter API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "qdrant_cloud_url" {
  description = "Qdrant Cloud cluster URL"
  type        = string
  sensitive   = true
  default     = ""
}

variable "qdrant_api_key" {
  description = "Qdrant Cloud API key"
  type        = string
  sensitive   = true
  default     = ""
}
