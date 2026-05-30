.PHONY: help install dev build up down logs shell test lint format clean deploy destroy

GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
APP_VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo unknown)
export GIT_SHA
export APP_VERSION

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv
	uv sync
	uv run python -m spacy download en_core_web_sm

dev: ## Run development server locally
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

ui: ## Run Streamlit UI
	uv run streamlit run ui.py

build: ## Build Docker images
	docker compose build


up: ## Start all services
	docker compose up

up-d: ## Start all services detached
	docker compose up -d

down: ## Stop all services
	docker compose down

logs: ## View logs from all services
	docker compose logs -f

shell: ## Open shell in agent-api container
	docker compose exec agent-api /bin/bash

test: ## Run tests
	uv run pytest

lint: ## Run linter
	uv run ruff check .

format: ## Format code
	uv run ruff format .

deploy: ## Deploy to GCP Cloud Run via Terraform
	cd terraform/gcp && terraform apply \
		-var "git_sha=$(GIT_SHA)" \
		-var "app_version=$(APP_VERSION)" \
		-replace=docker_image.app \
		-replace=docker_registry_image.app

destroy: ## Tear down GCP Cloud Run deployment
	cd terraform/gcp && terraform destroy

clean: ## Remove containers, volumes, and cache
	docker compose down -v
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
