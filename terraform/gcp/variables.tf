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

variable "qdrant_cloud_url" {
  description = "Qdrant Cloud cluster URL"
  type        = string
  default     = ""
}
