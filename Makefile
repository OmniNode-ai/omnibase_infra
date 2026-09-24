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
# Laptop profile (OMN-19496) -- the stack plus your own runtime, no lab or ops
# secrets, under its own compose project `omnibase-infra-local`:
#
#     make up-local            # write ~/.omnibase/local.env + model overlay if absent, then boot
#     make status-local        # migration gate, runtime /health bodies, delegate consumer group
#     make delegate-local PROMPT="..."  # one delegation through your runtime on the local broker
#     make down-local          # stop the laptop profile (keeps its volumes)
#     make down-local-volumes  # stop it and delete its volumes (local data)
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
        seed-keycloak seed-infisical _check-docker _check-env-file \
        local-env up-local status-local delegate-local down-local down-local-volumes

OMNIBASE_ENV_FILE ?= $(HOME)/.omnibase/.env
LOCAL_ENV_FILE ?= $(HOME)/.omnibase/local.env
LOCAL_OVERLAY_FILE ?= $(HOME)/.omnibase/local.bifrost.yaml
LOCAL_PROJECT := omnibase-infra-local
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

seed-infisical: _check-docker _check-env-file ## Seed Infisical from ONEX contracts (uses --execute)
	@uv run python scripts/seed-infisical.py \
	  --contracts-dir src/omnibase_infra/nodes \
	  --create-missing-keys \
	  --execute

# ----------------------------------------------------------------------------
# Laptop profile (OMN-19496): catalog bundle `local`, docker/catalog/bundles.yaml
# ----------------------------------------------------------------------------

local-env: ## Write the laptop env file and model overlay from their templates (never overwrites)
	@mkdir -p "$(dir $(LOCAL_ENV_FILE))" "$(dir $(LOCAL_OVERLAY_FILE))"
	@if [ -e "$(LOCAL_OVERLAY_FILE)" ]; then \
	  echo "==> Keeping existing model overlay $(LOCAL_OVERLAY_FILE)"; \
	else \
	  cp docker/lane-overlays/local.bifrost.example.yaml "$(LOCAL_OVERLAY_FILE)"; \
	  echo "==> Wrote model overlay $(LOCAL_OVERLAY_FILE)"; \
	fi
	@if [ -e "$(LOCAL_ENV_FILE)" ]; then \
	  echo "==> Keeping existing env file $(LOCAL_ENV_FILE)"; \
	else \
	  command -v openssl > /dev/null 2>&1 || { echo "ERROR: openssl is required to generate the local passwords"; exit 1; }; \
	  umask 077; \
	  sed -e "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$$(openssl rand -hex 32)|" \
	      -e "s|^VALKEY_PASSWORD=.*|VALKEY_PASSWORD=$$(openssl rand -hex 32)|" \
	      -e "s|^ONEX_LOCAL_BIFROST_OVERLAY=.*|ONEX_LOCAL_BIFROST_OVERLAY=$(LOCAL_OVERLAY_FILE)|" \
	      docker/local.env.example > "$(LOCAL_ENV_FILE)"; \
	  echo "==> Wrote env file $(LOCAL_ENV_FILE) (passwords generated)"; \
	fi
	@echo "==> Model endpoint: the line marked model_endpoint in $(LOCAL_OVERLAY_FILE)"

up-local: _check-docker local-env ## Laptop profile: build the runtime image and boot the stack + your runtime
	@echo "==> Starting the laptop profile (compose project $(LOCAL_PROJECT))..."
	$(ONEX_CLI) up local --env-file "$(LOCAL_ENV_FILE)" --build
	@echo "==> Started. A cold runtime takes several minutes to report healthy; run 'make status-local'."

status-local: _check-docker ## Laptop profile: migration gate, runtime /health bodies, delegate consumer group
	@echo "migration-gate: $$(docker inspect --format '{{.State.Health.Status}}' $(LOCAL_PROJECT)-migration-gate)"
	@echo "runtime main /health:";    curl -sS --max-time 10 http://localhost:8085/health; echo
	@echo "runtime effects /health:"; curl -sS --max-time 10 http://localhost:8086/health; echo
	@echo "delegate-skill command topic consumer groups:"
	@docker exec $(LOCAL_PROJECT)-redpanda rpk group list \
	  | awk 'NR==1 || /node_delegate_skill_orchestrator/' || true

delegate-local: _check-docker ## Laptop profile: one delegation through your runtime on the local broker (PROMPT="...")
	@test -n "$(PROMPT)" || { echo 'usage: make delegate-local PROMPT="Reply with exactly one word: hello"'; exit 2; }
	docker exec $(LOCAL_PROJECT)-runtime-effects onex delegate "$(PROMPT)" \
	  --bus kafka --kafka-bootstrap redpanda:9092 --locus deployed-lane

down-local: _check-docker ## Laptop profile: stop it (keeps its volumes)
	$(ONEX_CLI) down

down-local-volumes: _check-docker ## Laptop profile: stop it and delete its volumes (local data)
	$(ONEX_CLI) down --volumes

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
	  echo "Create it from the template and fill the required keys:"; \
	  echo "    KEYCLOAK_ADMIN_USERNAME=admin"; \
	  echo "    KEYCLOAK_ADMIN_PASSWORD=<secure-random>"; \
	  echo "    # plus any Infisical variables required by seed-infisical"; \
	  echo ""; \
	  echo "See .env.example for the full required key list."; \
	  echo "Generate secure passwords: openssl rand -base64 24"; \
	  echo "Or copy the template:       cp .env.example $(OMNIBASE_ENV_FILE)"; \
	  echo "Then edit:                  \$$EDITOR $(OMNIBASE_ENV_FILE)"; \
	  exit 1; \
	fi
