.PHONY: help install dev build up down logs shell test lint format clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv
	uv sync

dev: ## Run development server locally
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

build: ## Build Docker images
	docker compose build

up: ## Start all services
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

clean: ## Remove containers, volumes, and cache
	docker compose down -v
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
