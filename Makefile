.PHONY: help install dev dev-compose build up down restart logs shell test eval eval-retrieval eval-graph eval-graph-contract eval-rerank-sweep lint format clean deploy destroy

GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
APP_VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo unknown)
export GIT_SHA
export APP_VERSION

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv
	uv sync

dev: ## Run development server locally
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

dev-compose: ## Run API, UI, Qdrant, and Jaeger with container hot reload
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up

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

restart: ## Restart 
	docker compose restart

logs: ## View logs from all services
	docker compose logs -f

shell: ## Open shell in agent-api container
	docker compose exec agent-api /bin/bash

test: ## Run tests
	uv run pytest

eval: ## Run the full RAG eval: retrieval + generation + judges (needs Qdrant + API keys)
	uv run python -m evals.run_eval --full

eval-retrieval: ## Run retrieval + embedding checks only, no LLM (what CI runs on push to main)
	uv run python -m evals.run_eval

eval-graph: ## Manually evaluate live graph outcomes and citations against the eval corpus
	uv run python -m evals.run_graph_eval

eval-graph-contract: ## Run deterministic graph workflow contracts (no services or API keys)
	uv run pytest tests/integration/test_agent_graph.py

eval-rerank-sweep: ## Calibrate RERANK_SCORE_FLOOR on the golden set (needs Qdrant + OpenAI key)
	uv run python -m evals.rerank_sweep

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
