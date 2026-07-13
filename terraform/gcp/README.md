# GCP deployment

Terraform creates Cloud Run, Cloud SQL, and the Secret Manager containers the service needs.
Provider keys are never stored in Terraform; copy the example configuration and set the project
and Qdrant endpoint. Bootstrap the three externally managed provider secrets once:

```sh
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform apply -target=google_secret_manager_secret.runtime
```

Then seed one version for each secret:

```sh
gcloud secrets versions add doc-research-agent-openai-api-key --data-file=-
gcloud secrets versions add doc-research-agent-openrouter-api-key --data-file=-
gcloud secrets versions add doc-research-agent-qdrant-api-key --data-file=-
```

Replace `doc-research-agent` when `service_name` differs. The commands read each value from
standard input; do not add provider keys to `terraform.tfvars`.
If an existing `terraform.tfvars` has `openai_api_key`, `openrouter_api_key`, or
`qdrant_api_key`, remove those obsolete entries before planning.

Finally, plan and apply the service:

```sh
terraform plan
terraform apply
```

Terraform generates the database password ephemerally and writes it directly to Cloud SQL and a
dedicated Secret Manager version. Neither the password nor the database URL is stored in state.
Cloud Run uses its dedicated runtime service account with access only to these secrets and the
Cloud SQL client role. The Cloud SQL instance has automatic backups, point-in-time recovery, and
deletion protection; removing it requires an explicit configuration change first.
