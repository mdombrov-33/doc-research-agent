# GCP deployment

Terraform creates the three Secret Manager containers Cloud Run needs, but never stores secret
values. Copy the example configuration and set the project and Qdrant endpoint. Then bootstrap
the empty containers once:

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

The Cloud Run service uses its dedicated runtime service account. It has `secretAccessor` only
on these three secrets.
