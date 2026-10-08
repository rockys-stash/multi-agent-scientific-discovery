.PHONY: help setup test test-ui lint format web serve e1 e2 e3 e4 e5 experiments clean

help: ## List targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install Python (uv) and console (npm) dependencies
	uv sync --locked
	npm --prefix web ci

test: ## Unit, workflow, experiment-harness and API tests (fixtures only, offline)
	uv run pytest

test-ui: web ## Browser, keyboard and axe accessibility tests (needs Chromium)
	uv run pytest -m ui

lint: ## Ruff, mypy, ESLint and TypeScript checks
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests
	npm --prefix web run lint
	npm --prefix web run typecheck

format: ## Auto-format Python
	uv run ruff format .
	uv run ruff check --fix .

web: ## Build the console into web/dist
	npm --prefix web run build

serve: web ## Serve the console and API on http://127.0.0.1:8000 (reads .env if present)
	set -a; [ -f .env ] && . ./.env; set +a; uv run discoverylab serve

e1: ## E1 end-to-end runs (needs the scholarly APIs; model runs need ANTHROPIC_API_KEY)
	set -a; [ -f .env ] && . ./.env; set +a; uv run discoverylab experiment e1_runs
e2: ## E2 verifier validity (on E1's verified citations)
	uv run discoverylab experiment e2_verifier
e3: ## E3 critic fault injection (offline, on E1's runs)
	uv run discoverylab experiment e3_critic
e4: ## E4 fluency against process (judge needs ANTHROPIC_API_KEY)
	set -a; [ -f .env ] && . ./.env; set +a; uv run discoverylab experiment e4_fluency
e5: ## E5 reproducibility (offline replay of E1's runs)
	uv run discoverylab experiment e5_reproducibility

experiments: e1 e2 e3 e4 e5 ## Run E1-E5 in order

clean: ## Remove build output and caches (keeps runs/, cache/ and results/)
	rm -rf web/dist .pytest_cache .mypy_cache .ruff_cache
