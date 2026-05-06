# omnibase_infra Makefile — user-facing infra entrypoints
#
# Provides in-repo runnable commands so users can do
#
#     cd repos/omnibase_infra && make up
#
# instead of needing to know the bundle CLI path. All targets that touch Docker
# detect a missing/stopped daemon and emit an actionable error before doing
# anything destructive. See OMN-10377 for the architectural rationale (Docker
# orchestration must live in this repo, not in the public `omnibase` shell).
#
# Usage:
#
#     make help            # List all targets
#     make up              # Start core infra bundle (postgres, redpanda, valkey, infisical)
#     make up-auth         # Start the auth bundle (keycloak)
#     make up-runtime      # Start the runtime bundle (depends on core)
#     make down            # Stop the core bundle ONLY (auth/runtime stay running)
#     make down-auth       # Stop the auth bundle
#     make down-runtime    # Stop the runtime bundle
#     make down-all        # Stop runtime, then auth, then core (full teardown)
#     make status          # Show running omnibase-infra containers
#     make seed-keycloak   # Reconcile Keycloak clients from desired-clients.json
#     make seed-infisical  # Seed Infisical from ONEX contracts (writes with --execute)
#
# Environment:
#
#     OMNIBASE_ENV_FILE   Override env file path (default: ~/.omnibase/.env)
#     KC_URL              Keycloak base URL for seed-keycloak (default: http://localhost:28080)
#     KC_REALM            Realm to seed (default: omninode)
#
# All `up*` targets delegate to the catalog CLI documented in CLAUDE.md:
#     uv run python -m omnibase_infra.docker.catalog.cli up <bundle>
# `seed-keycloak` delegates to scripts/seed-keycloak.sh (PR #1500).
# `seed-infisical` delegates to scripts/seed-infisical.py.

.PHONY: help up up-auth up-runtime down down-auth down-runtime down-all status \
        seed-keycloak seed-infisical _check-docker _check-env-file

OMNIBASE_ENV_FILE ?= $(HOME)/.omnibase/.env
ONEX_CLI := uv run python -m omnibase_infra.docker.catalog.cli

help: ## Show this help
	@echo "omnibase_infra targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(firstword $(MAKEFILE_LIST)) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

up: _check-docker ## Start core infra bundle (postgres, redpanda, valkey, infisical)
	@echo "==> Starting core infrastructure bundle..."
	$(ONEX_CLI) up core
	@echo "==> Done. Run 'make status' to verify, or 'make up-auth' to add Keycloak."

up-auth: _check-docker ## Start the auth bundle (keycloak); core must be up first
	@echo "==> Starting auth (keycloak) bundle..."
	$(ONEX_CLI) up auth
	@echo "==> Done. Run 'make seed-keycloak' to reconcile clients."

up-runtime: _check-docker ## Start the full runtime bundle (extends core)
	@echo "==> Starting runtime bundle..."
	$(ONEX_CLI) up runtime

down: _check-docker ## Stop the core bundle ONLY (use down-all for full teardown)
	@echo "==> Stopping core infrastructure bundle..."
	$(ONEX_CLI) down core

down-auth: _check-docker ## Stop the auth bundle (keycloak)
	@echo "==> Stopping auth (keycloak) bundle..."
	$(ONEX_CLI) down auth

down-runtime: _check-docker ## Stop the runtime bundle
	@echo "==> Stopping runtime bundle..."
	$(ONEX_CLI) down runtime

down-all: _check-docker ## Stop runtime, then auth, then core (full teardown)
	@echo "==> Stopping runtime bundle (if running)..."
	-$(ONEX_CLI) down runtime
	@echo "==> Stopping auth bundle (if running)..."
	-$(ONEX_CLI) down auth
	@echo "==> Stopping core bundle (if running)..."
	-$(ONEX_CLI) down core
	@echo "==> Done. All omnibase-infra bundles stopped."

status: _check-docker ## Show running omnibase-infra containers
	@docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' \
	  | awk 'NR==1 || /omnibase-infra-/' \
	  || true

seed-keycloak: _check-docker _check-env-file ## Reconcile Keycloak clients from desired-clients.json
	@bash scripts/seed-keycloak.sh

seed-infisical: _check-env-file ## Seed Infisical from ONEX contracts (uses --execute)
	@uv run python scripts/seed-infisical.py \
	  --contracts-dir src/omnibase_infra/nodes \
	  --create-missing-keys \
	  --execute

# ----------------------------------------------------------------------------
# Internal helpers (not part of the public target surface)
# ----------------------------------------------------------------------------

# Detect a missing or stopped Docker daemon and emit an actionable error rather
# than letting `docker compose ...` fail with a cryptic message. omnibase_infra
# is the boundary where Docker becomes a hard requirement (per OMN-10377 /
# OMN-10378); the public `omnibase` repo never assumes it.
_check-docker:
	@if ! command -v docker > /dev/null 2>&1; then \
	  echo "ERROR: docker is not installed."; \
	  echo "  Install from https://docs.docker.com/get-docker/"; \
	  echo "  macOS:  brew install --cask docker"; \
	  echo "  Linux:  see https://docs.docker.com/engine/install/"; \
	  exit 1; \
	fi
	@if ! docker info > /dev/null 2>&1; then \
	  echo "ERROR: docker daemon is not running."; \
	  echo "  macOS:  open -a Docker"; \
	  echo "  Linux:  sudo systemctl start docker"; \
	  exit 1; \
	fi

# Check that ~/.omnibase/.env (or whatever OMNIBASE_ENV_FILE points to) exists.
# Fail with a clear remediation path rather than a stack trace from a
# downstream script.
_check-env-file:
	@if [ ! -f "$(OMNIBASE_ENV_FILE)" ]; then \
	  echo "ERROR: env file not found at $(OMNIBASE_ENV_FILE)"; \
	  echo ""; \
	  echo "Create it with at minimum:"; \
	  echo "    KEYCLOAK_ADMIN_USERNAME=admin"; \
	  echo "    KEYCLOAK_ADMIN_PASSWORD=<secure-random>"; \
	  echo ""; \
	  echo "Generate a secure password: openssl rand -base64 24"; \
	  echo "Or copy the template:       cp .env.example $(OMNIBASE_ENV_FILE)"; \
	  echo "Then edit:                  \$$EDITOR $(OMNIBASE_ENV_FILE)"; \
	  exit 1; \
	fi
