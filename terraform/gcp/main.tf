terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "docker" {
  registry_auth {
    address  = "${var.region}-docker.pkg.dev"
    username = "oauth2accesstoken"
    password = data.google_client_config.default.access_token
  }
}

data "google_client_config" "default" {}

resource "google_project_service" "cloudrun" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudbuild" {
  service            = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "secretmanager" {
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

locals {
  runtime_secret_ids = {
    openai_api_key     = "${var.service_name}-openai-api-key"
    openrouter_api_key = "${var.service_name}-openrouter-api-key"
    qdrant_api_key     = "${var.service_name}-qdrant-api-key"
  }
}

resource "google_service_account" "runtime" {
  account_id   = "${var.service_name}-runtime"
  display_name = "${var.service_name} Cloud Run runtime"
}

resource "google_secret_manager_secret" "runtime" {
  for_each  = local.runtime_secret_ids
  secret_id = each.value

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_iam_member" "runtime" {
  for_each  = google_secret_manager_secret.runtime
  project   = var.project_id
  secret_id = each.value.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_artifact_registry_repository" "app" {
  location      = var.region
  repository_id = var.service_name
  format        = "DOCKER"
  description   = "Docker repository for ${var.service_name}"
}

resource "docker_image" "app" {
  name = "${var.region}-docker.pkg.dev/${var.project_id}/${var.service_name}/${var.service_name}:${var.docker_image_tag}"

  build {
    context    = "${path.module}/../.."
    dockerfile = "Dockerfile"
    platform   = "linux/amd64"
    no_cache   = true
    build_args = {
      GIT_SHA     = var.git_sha
      APP_VERSION = var.app_version
    }
  }

  depends_on = [
    google_project_service.cloudbuild,
    google_artifact_registry_repository.app
  ]
}

resource "docker_registry_image" "app" {
  name = docker_image.app.name

  depends_on = [
    google_artifact_registry_repository.app,
    docker_image.app
  ]
}

resource "google_cloud_run_service" "app" {
  name     = var.service_name
  location = var.region

  template {
    spec {
      service_account_name = google_service_account.runtime.email

      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.service_name}/${var.service_name}@${docker_registry_image.app.sha256_digest}"

        resources {
          limits = {
            cpu    = "1"
            memory = "2Gi"
          }
        }

        env {
          name = "OPENAI_API_KEY"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.runtime["openai_api_key"].secret_id
              key  = "latest"
            }
          }
        }
        env {
          name = "OPENROUTER_API_KEY"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.runtime["openrouter_api_key"].secret_id
              key  = "latest"
            }
          }
        }
        env {
          name  = "QDRANT_MODE"
          value = "cloud"
        }
        env {
          name  = "QDRANT_CLOUD_URL"
          value = var.qdrant_cloud_url
        }
        env {
          name = "QDRANT_API_KEY"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.runtime["qdrant_api_key"].secret_id
              key  = "latest"
            }
          }
        }
        env {
          name  = "QDRANT_COLLECTION_NAME"
          value = "documents"
        }
        env {
          name  = "EMBEDDING_MODEL"
          value = "text-embedding-3-small"
        }
        env {
          name  = "EMBEDDING_DIMENSION"
          value = "1536"
        }
        env {
          name  = "APP_ENV"
          value = "production"
        }
        env {
          name  = "LOG_LEVEL"
          value = "INFO"
        }
        env {
          name  = "UPLOAD_DIR"
          value = "./uploads"
        }
        env {
          name  = "HOST"
          value = "0.0.0.0"
        }
        env {
          name  = "PYTHONUNBUFFERED"
          value = "1"
        }
        env {
          name  = "GIT_SHA"
          value = var.git_sha
        }
        env {
          name  = "APP_VERSION"
          value = var.app_version
        }

        ports {
          container_port = 8000
        }
      }
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = "0"
        "autoscaling.knative.dev/maxScale" = "1"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [
    google_project_service.cloudrun,
    docker_registry_image.app,
    google_secret_manager_secret_iam_member.runtime,
  ]
}

resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.app.name
  location = google_cloud_run_service.app.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "service_url" {
  value       = google_cloud_run_service.app.status[0].url
  description = "URL of the deployed Cloud Run service"
}
