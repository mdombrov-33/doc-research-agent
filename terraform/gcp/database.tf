module "postgres" {
  source  = "terraform-google-modules/sql-db/google//modules/postgresql"
  version = "28.1.0"

  project_id          = var.project_id
  region              = var.region
  name                = "${var.service_name}-postgres"
  database_version    = "POSTGRES_16"
  edition             = "ENTERPRISE"
  tier                = "db-custom-1-3840"
  db_name             = local.database_name
  enable_default_user = false

  backup_configuration = {
    enabled                        = true
    point_in_time_recovery_enabled = true
    retained_backups               = 7
    retention_unit                 = "COUNT"
    transaction_log_retention_days = "7"
  }
  connector_enforcement       = true
  deletion_protection         = true
  deletion_protection_enabled = true
  disk_size                   = 10
  password_validation_policy_config = {
    min_length                  = 24
    complexity                  = "COMPLEXITY_DEFAULT"
    disallow_username_substring = true
  }
  ip_configuration = {
    ipv4_enabled = true
    ssl_mode     = "ENCRYPTED_ONLY"
  }
  user_labels = {
    service = var.service_name
  }

  module_depends_on = [
    google_project_service.compute,
    google_project_service.sqladmin,
  ]
}

ephemeral "random_password" "database" {
  length           = 32
  min_lower        = 1
  min_numeric      = 1
  min_special      = 1
  min_upper        = 1
  special          = true
  override_special = "!-._~"
}

resource "google_sql_user" "app" {
  name                = local.database_user
  instance            = module.postgres.instance_name
  password_wo         = ephemeral.random_password.database.result
  password_wo_version = 1
  deletion_policy     = "ABANDON"
}

resource "google_secret_manager_secret_version" "database_url" {
  secret                 = google_secret_manager_secret.runtime["database_url"].id
  secret_data_wo         = "postgresql://${local.database_user}:${urlencode(ephemeral.random_password.database.result)}@/${local.database_name}?host=/cloudsql/${module.postgres.instance_connection_name}"
  secret_data_wo_version = 1
  deletion_policy        = "ABANDON"

  depends_on = [google_sql_user.app]
}
