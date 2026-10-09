.PHONY: help setup setup-local models test test-ui lint format web serve e1 e2 e3 e4 e5 experiments results screenshots clean

help: ## List targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install Python (uv) and console (npm) dependencies
	uv sync --locked
	npm --prefix web ci

setup-local: ## Also install llama-cpp-python for the local open-weights models (compiles; a few minutes)
	uv sync --locked --group local

QWEN = https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_0.gguf
PHI = https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf

models: ## Download the local reasoner and judge weights (about 4.9 GB) into models/ and check their hashes
	mkdir -p models
	[ -f models/qwen3-4b-instruct-2507-q4_0.gguf ] || curl -fL -o models/qwen3-4b-instruct-2507-q4_0.gguf $(QWEN)
	[ -f models/phi-4-mini-instruct-q4_k_m.gguf ] || curl -fL -o models/phi-4-mini-instruct-q4_k_m.gguf $(PHI)
	cd models && sha256sum -c ../configs/models.sha256

test: ## Unit, workflow, experiment-harness and API tests (fixtures only, offline)
	uv run pytest

test-ui: web ## Browser, keyboard and axe accessibility tests (needs Chromium)
	uv run pytest -m ui

lint: ## Ruff, mypy, ESLint and TypeScript checks
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests scripts
	npm --prefix web run lint
	npm --prefix web run typecheck

format: ## Auto-format Python
	uv run ruff format .
	uv run ruff check --fix .

web: ## Build the console into web/dist
	npm --prefix web run build

serve: web ## Serve the console and API on http://127.0.0.1:8000 (reads .env if present)
	set -a; [ -f .env ] && . ./.env; set +a; uv run discoverylab serve

e1: ## E1 end-to-end runs (needs the scholarly APIs; local model runs need `make setup-local models`)
	set -a; [ -f .env ] && . ./.env; set +a; uv run discoverylab experiment e1_runs
e2: ## E2 verifier validity (on E1's verified citations)
	uv run discoverylab experiment e2_verifier
e3: ## E3 critic fault injection (offline, on E1's runs; model critic needs the local model)
	set -a; [ -f .env ] && . ./.env; set +a; uv run discoverylab experiment e3_critic
e4: ## E4 fluency against process (judge needs the local judge model)
	set -a; [ -f .env ] && . ./.env; set +a; uv run discoverylab experiment e4_fluency
e5: ## E5 reproducibility (offline replay of E1's runs)
	set -a; [ -f .env ] && . ./.env; set +a; uv run discoverylab experiment e5_reproducibility

experiments: e1 e2 e3 e4 e5 results ## Run E1-E5 in order, then render docs/generated/results.md

screenshots: web ## Capture console screenshots from the real runs into docs/screenshots
	uv run python scripts/screenshots.py

results: ## Render the latest result of every experiment to docs/generated/results.md
	uv run discoverylab results

clean: ## Remove build output and caches (keeps runs/, cache/ and results/)
	rm -rf web/dist .pytest_cache .mypy_cache .ruff_cache
