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
#     make down            # Stop the last generated compose project (keeps volumes)
#     make down-auth       # Stop the last generated compose project (bundle argument ignored; keeps volumes)
#     make down-runtime    # Stop the last generated compose project (bundle argument ignored; keeps volumes)
#     make down-all        # Stop the last generated compose project (keeps volumes; repeats down for legacy arguments)
#     make status          # Show running omnibase-infra containers
#     make seed-keycloak   # Reconcile Keycloak clients from desired-clients.json
#     make seed-infisical  # Seed Infisical from ONEX contracts (writes with --execute)
#
# Laptop profile (OMN-19496) -- the stack plus your own runtime, no lab or ops
# secrets, under its own compose project `omnibase-infra-local`. The projection
# API runs in `local-market` on the omnimarket image (OMN-21279).
#
#     make up-local            # write ~/.omnibase/local.env + model overlay if absent, then boot
#     make status-local        # migration gate, runtime /health bodies, delegate consumer group
#     make check-sasl-local    # prove the local broker refuses anonymous clients and accepts onex-local (OMN-20820)
#     make mint-local          # mint this machine's login on the local broker as lane local, then prove it (OMN-20823)
#     make delegate-local PROMPT="..."  # one delegation through your runtime on the local broker
#     make tenant-local        # give your runtime a tenant (delegation refuses without one), then make up-local
#     make secret-local PROVIDER=gemini # register your tenant's provider key from stdin or a hidden prompt
#     make up-local-mirror     # up-local plus the mirror-only forwarder to the lab lane your overlay names (OMN-19983)
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
# All non-local `down*` targets stop the whole last generated compose project;
# cmd_down ignores bundle arguments and keeps volumes unless --volumes is passed.
# `seed-keycloak` delegates to scripts/seed-keycloak.sh (PR #1500).
# `seed-infisical` delegates to scripts/seed-infisical.py.

.PHONY: help up up-auth up-runtime down down-auth down-runtime down-all status \
        seed-keycloak seed-infisical _check-docker _check-env-file \
        local-env up-local up-local-mirror status-local check-sasl-local mint-local delegate-local tenant-local secret-local down-local down-local-volumes

OMNIBASE_ENV_FILE ?= $(HOME)/.omnibase/.env
LOCAL_ENV_FILE ?= $(HOME)/.omnibase/local.env
LOCAL_OVERLAY_FILE ?= $(HOME)/.omnibase/local.bifrost.yaml
LOCAL_PROJECT := omnibase-infra-local
# OMN-20820: the one SCRAM superuser redpanda-local-sasl creates on the laptop broker.
LOCAL_SASL_USER := onex-local
# OMN-20823: make mint-local. The Admin API as runtime-effects reaches it, the grants
# file of the developer principal, and the one prompt the host-side probe sends.
LOCAL_ADMIN_URL ?= http://redpanda:9644
MINT_DECLARATION := deploy/local/developer-principal-grants.yaml
MINT_PROBE_PROMPT := Reply with exactly one word: hello
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

down: _check-docker ## Stop the last generated compose project (keeps volumes)
	@echo "==> Stopping the last generated compose project (keeps volumes)..."
	$(ONEX_CLI) down core

down-auth: _check-docker ## Stop the last generated compose project (bundle argument ignored; keeps volumes)
	@echo "==> Stopping the last generated compose project (keeps volumes)..."
	$(ONEX_CLI) down auth

down-runtime: _check-docker ## Stop the last generated compose project (bundle argument ignored; keeps volumes)
	@echo "==> Stopping the last generated compose project (keeps volumes)..."
	$(ONEX_CLI) down runtime

down-all: _check-docker ## Stop the last generated compose project (keeps volumes; repeats down for legacy arguments)
	@echo "==> Stopping the last generated compose project (legacy runtime argument)..."
	-$(ONEX_CLI) down runtime
	@echo "==> Stopping the last generated compose project (legacy auth argument)..."
	-$(ONEX_CLI) down auth
	@echo "==> Stopping the last generated compose project (legacy core argument)..."
	-$(ONEX_CLI) down core
	@echo "==> Done. Last generated compose project stopped; volumes kept."

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
	@command -v openssl > /dev/null 2>&1 || { echo "ERROR: openssl is required to generate the local passwords"; exit 1; }
	@if [ -e "$(LOCAL_ENV_FILE)" ]; then \
	  echo "==> Keeping existing env file $(LOCAL_ENV_FILE)"; \
	  for var in OMNINODE_RUNTIME_PASSWORD TENANT_PROJECTION_WRITER_PASSWORD REDPANDA_SUPERUSER_PASSWORD; do \
	    grep -q "^$$var=" "$(LOCAL_ENV_FILE)" || { \
	      chmod 600 "$(LOCAL_ENV_FILE)" || exit 1; \
	      [ -z "$$(tail -c 1 "$(LOCAL_ENV_FILE)")" ] || echo >> "$(LOCAL_ENV_FILE)"; \
	      printf '%s=%s\n' "$$var" "$$(openssl rand -hex 32)" >> "$(LOCAL_ENV_FILE)"; \
	      echo "==> Added $$var to $(LOCAL_ENV_FILE) (generated)"; \
	    }; \
	  done; \
	else \
	  umask 077; \
	  sed -e "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$$(openssl rand -hex 32)|" \
	      -e "s|^VALKEY_PASSWORD=.*|VALKEY_PASSWORD=$$(openssl rand -hex 32)|" \
	      -e "s|^OMNINODE_RUNTIME_PASSWORD=.*|OMNINODE_RUNTIME_PASSWORD=$$(openssl rand -hex 32)|" \
	      -e "s|^TENANT_PROJECTION_WRITER_PASSWORD=.*|TENANT_PROJECTION_WRITER_PASSWORD=$$(openssl rand -hex 32)|" \
	      -e "s|^REDPANDA_SUPERUSER_PASSWORD=.*|REDPANDA_SUPERUSER_PASSWORD=$$(openssl rand -hex 32)|" \
	      -e "s|^ONEX_LOCAL_BIFROST_OVERLAY=.*|ONEX_LOCAL_BIFROST_OVERLAY=$(LOCAL_OVERLAY_FILE)|" \
	      docker/local.env.example > "$(LOCAL_ENV_FILE)"; \
	  echo "==> Wrote env file $(LOCAL_ENV_FILE) (passwords generated)"; \
	fi
	@echo "==> Model endpoint: the line marked model_endpoint in $(LOCAL_OVERLAY_FILE)"

up-local: _check-docker local-env ## Laptop profile: build the runtime image and boot the stack + your runtime
	@echo "==> Starting the laptop profile (compose project $(LOCAL_PROJECT))..."
	$(ONEX_CLI) up local --env-file "$(LOCAL_ENV_FILE)" --build
	@echo "==> Started. A cold runtime takes several minutes to report healthy; run 'make status-local'."

up-local-mirror: _check-docker local-env ## Laptop profile plus the mirror-only forwarder to the lab lane your overlay names (OMN-19983)
	@test -f "$(LOCAL_ENV_FILE)" || { echo "ERROR: $(LOCAL_ENV_FILE) is missing; run make local-env." >&2; exit 2; }
	@for var in ONEX_LOCAL_LANE_MIRROR_OVERLAY ONEX_LOCAL_LANE_CREDENTIAL_MAP; do \
	  grep -q "^$$var=." "$(LOCAL_ENV_FILE)" || { echo "ERROR: $$var is not set in $(LOCAL_ENV_FILE); see docker/local.env.example" >&2; exit 2; }; \
	done
	@echo "==> Starting the laptop profile with the lane mirror (compose project $(LOCAL_PROJECT))..."
	$(ONEX_CLI) up local-lane-mirror --env-file "$(LOCAL_ENV_FILE)" --build

status-local: _check-docker ## Laptop profile: migration gate, runtime /health bodies, delegate consumer group
	@echo "migration-gate: $$(docker inspect --format '{{.State.Health.Status}}' $(LOCAL_PROJECT)-migration-gate)"
	@echo "runtime main /health:";    curl -sS --max-time 10 http://localhost:8085/health; echo
	@echo "runtime effects /health:"; curl -sS --max-time 10 http://localhost:8086/health; echo
	@echo "delegate-skill command topic consumer groups:"
	@test -f "$(LOCAL_ENV_FILE)" || { echo "ERROR: $(LOCAL_ENV_FILE) is missing; run make local-env." >&2; exit 2; }
	@pass=$$(sed -n 's/^REDPANDA_SUPERUSER_PASSWORD=//p' "$(LOCAL_ENV_FILE)"); \
	  test -n "$$pass" || { echo 'ERROR: REDPANDA_SUPERUSER_PASSWORD is absent from $(LOCAL_ENV_FILE); run make local-env.' >&2; exit 2; }; \
	  groups=$$(RPK_PASS="$$pass" docker exec -e RPK_USER=$(LOCAL_SASL_USER) -e RPK_PASS -e RPK_SASL_MECHANISM=SCRAM-SHA-256 \
	    $(LOCAL_PROJECT)-redpanda rpk group list) || exit 1; \
	  printf '%s\n' "$$groups" | awk 'NR==1 || /node_delegate_skill_orchestrator/'

check-sasl-local: _check-docker ## Laptop profile: prove the broker refuses anonymous clients and accepts the local superuser (OMN-20820)
	@test -f "$(LOCAL_ENV_FILE)" || { echo "ERROR: $(LOCAL_ENV_FILE) is missing; run make local-env." >&2; exit 2; }
	@pass=$$(sed -n 's/^REDPANDA_SUPERUSER_PASSWORD=//p' "$(LOCAL_ENV_FILE)"); \
	  test -n "$$pass" || { echo 'ERROR: REDPANDA_SUPERUSER_PASSWORD is absent from $(LOCAL_ENV_FILE); run make local-env.' >&2; exit 2; }; \
	  broker=$(LOCAL_PROJECT)-redpanda; \
	  if anon=$$(docker exec $$broker rpk cluster info -X brokers=localhost:19092 2>&1); then \
	    echo "FAIL: an anonymous client was accepted by the local broker (SASL is not enforced)" >&2; exit 1; \
	  fi; \
	  printf '%s\n' "$$anon" | grep -q 'SASL is required but not provided' || { \
	    echo "FAIL: the anonymous client was refused, but not for want of SASL:" >&2; printf '%s\n' "$$anon" >&2; exit 1; }; \
	  RPK_PASS="$$pass" docker exec -e RPK_USER=$(LOCAL_SASL_USER) -e RPK_PASS -e RPK_SASL_MECHANISM=SCRAM-SHA-256 \
	    $$broker rpk cluster info -X brokers=localhost:19092 > /dev/null || { \
	    echo "FAIL: the local superuser $(LOCAL_SASL_USER) was refused" >&2; exit 1; }; \
	  echo "==> OK: the local broker refuses anonymous clients and accepts $(LOCAL_SASL_USER)"

# OMN-20823 (plan D6). The lock directory beside the env file is taken before
# anything else and released by the trap only while it still holds our pid; a
# lock whose recorded process is gone is removed and said, under a second guard
# directory so two runs cannot both recover it. The password is generated here,
# reaches the node only as the variable ONEX_MINT_PASSWORD of docker exec (never
# an argument), reaches lane-login only on its stdin, and is never printed. The
# state-root search reads it from its own environment: exit 0 clean, 10 found,
# anything else fails closed. `onex node --output receipt` prints one JSON whose
# .result.handler_result is the orchestrator's result.
mint-local: _check-docker ## Laptop profile: mint this machine's login on the local broker and prove it with one delegation (OMN-20823)
	@lock="$$(dirname "$(LOCAL_ENV_FILE)")/.mint-local.lock"; guard="$$lock.recover"; \
	  mkdir -p "$$(dirname "$$lock")" || exit 1; \
	  busy() { echo "mint-local: busy, lock held by pid $$1 (exit 75)" >&2; exit 75; }; \
	  if ! mkdir "$$lock" 2>/dev/null; then \
	    holder=$$(cat "$$lock/pid" 2>/dev/null); \
	    if [ -n "$$holder" ] && kill -0 "$$holder" 2>/dev/null; then busy "$$holder"; fi; \
	    if [ -z "$$holder" ] && [ -n "$$(find "$$lock" -maxdepth 0 -mmin -1 2>/dev/null)" ]; then busy unknown; fi; \
	    if ! mkdir "$$guard" 2>/dev/null; then \
	      [ -n "$$(find "$$guard" -maxdepth 0 -mmin +1 2>/dev/null)" ] && rmdir "$$guard" 2>/dev/null; \
	      busy "$${holder:-unknown}"; \
	    fi; \
	    now=$$(cat "$$lock/pid" 2>/dev/null); \
	    if [ "$$now" = "$$holder" ] && ! { [ -n "$$now" ] && kill -0 "$$now" 2>/dev/null; }; then \
	      echo "mint-local: removing a stale lock left by pid $${holder:-unknown}, which is not running"; \
	      rm -rf "$$lock"; \
	    fi; \
	    mkdir "$$lock" 2>/dev/null && echo "$$$$" > "$$lock/pid"; got=$$?; rmdir "$$guard"; \
	    [ "$$got" -eq 0 ] || busy "$$(cat "$$lock/pid" 2>/dev/null || echo unknown)"; \
	  fi; \
	  echo "$$$$" > "$$lock/pid"; \
	  work=""; container="$(LOCAL_PROJECT)-runtime-effects"; in_dir="/tmp/onex-mint-$$$$"; \
	  trap 'docker exec -u omniinfra "$$container" rm -rf "$$in_dir" >/dev/null 2>&1; [ "$$(cat "$$lock/pid" 2>/dev/null)" = "$$$$" ] && rm -rf "$$lock"; rm -rf $${work:+"$$work"}' EXIT; \
	  trap 'exit 130' INT; trap 'exit 143' TERM; \
	  work=$$(mktemp -d) || exit 1; \
	  $(MAKE) --no-print-directory check-sasl-local || { \
	    echo "mint-local: the local broker is not secured; run make up-local first" >&2; exit 1; }; \
	  user=$$(id -un) && host=$$(hostname -s) || exit 1; \
	  tenant=$$(sed -n 's/^ONEX_TENANT_ID=//p' "$(LOCAL_ENV_FILE)" | tail -n 1); \
	  scan='import os, pathlib, sys; s = os.environb.get(b"ONEX_MINT_PASSWORD", b""); r = pathlib.Path(sys.argv[1]); sys.exit(2) if not (s and r.is_dir()) else sys.exit(10 if any(s in p.read_bytes() for p in r.rglob("*") if p.is_file()) else 0)'; \
	  run_node() { \
	    jq -n --arg mode "$$1" --rawfile decl "$(MINT_DECLARATION)" --arg login "$$user" \
	      --arg device "$$host" --arg url "$(LOCAL_ADMIN_URL)" --arg pwenv "$$2" \
	      '{mode: $$mode, declaration_yaml: $$decl, login: $$login, device: $$device, admin_url: $$url, password_env: $$pwenv}' \
	      > "$$work/input.json" && chmod 644 "$$work/input.json" || return 1; \
	    docker exec -u omniinfra "$$container" mkdir -p "$$in_dir" \
	      && docker cp "$$work/input.json" "$$container:$$in_dir/input.json" > /dev/null || return 1; \
	    docker exec -u omniinfra -e HOME=/home/omniinfra -e ONEX_MINT_PASSWORD "$$container" \
	      onex node node_developer_principal_mint_orchestrator --input "$$in_dir/input.json" \
	      --state-root "$$in_dir/state" --output receipt > "$$work/receipt.json"; \
	    if [ -n "$$2" ]; then \
	      docker exec -u omniinfra -e ONEX_MINT_PASSWORD "$$container" python3 -c "$$scan" "$$in_dir/state"; \
	      case $$? in \
	        0) ;; \
	        10) echo "mint-local: the password was found under the onex node state root" >&2; return 1 ;; \
	        *) echo "mint-local: the onex node state root could not be searched for the password" >&2; return 1 ;; \
	      esac; \
	    fi; \
	    docker exec -u omniinfra "$$container" rm -rf "$$in_dir" || return 1; \
	    [ "$$(jq -r '.result.handler_result.status // empty' "$$work/receipt.json" 2>/dev/null)" = success ] && return 0; \
	    echo "mint-local: the mint stopped at $$(jq -r '.result.handler_result.failed_step // "the node"' "$$work/receipt.json" 2>/dev/null): $$(jq -r '.result.handler_result.reason // "no receipt"' "$$work/receipt.json" 2>/dev/null)" >&2; \
	    jq -r '.result.handler_result.extra_acls[]? | "mint-local: ACL outside the plan: " + .' "$$work/receipt.json" >&2 2>/dev/null; \
	    return 1; \
	  }; \
	  probe() { \
	    rm -rf "$$work/probe" && mkdir "$$work/probe" || return 1; \
	    if [ -n "$$tenant" ]; then \
	      ONEX_TENANT_ID="$$tenant" onex delegate --bus kafka --lane local --json --state-root "$$work/probe" \
	        "$(MINT_PROBE_PROMPT)" > "$$work/probe.json" 2> "$$work/probe.err"; \
	    else \
	      onex delegate --bus kafka --lane local --json --state-root "$$work/probe" \
	        "$(MINT_PROBE_PROMPT)" > "$$work/probe.json" 2> "$$work/probe.err"; \
	    fi; \
	    cid=$$(jq -r '.correlation_id // empty' "$$work/probe.json" 2>/dev/null); \
	    run=$$(jq -r '.run_id // empty' "$$work/probe.json" 2>/dev/null); \
	    [ "$$(jq -r '.status // empty' "$$work/probe.json" 2>/dev/null)" = success ] && [ -n "$$cid" ] && [ -n "$$run" ] \
	      && jq -e --arg cid "$$cid" '.terminal_payload.correlation_id == $$cid' \
	        "$$work/probe/runs/$$run/workflow_result.json" > /dev/null 2>&1; \
	  }; \
	  run_node audit "" || exit 1; \
	  principal=$$(jq -r '.result.handler_result.principal // empty' "$$work/receipt.json"); \
	  if [ "$$(jq -r '.result.handler_result | (.exists == true) and (.extra_acls | length) == 0 and (.missing_acls | length) == 0' "$$work/receipt.json")" = true ] \
	      && probe; then \
	    echo "mint-local: $$principal unchanged"; exit 0; \
	  fi; \
	  ONEX_MINT_PASSWORD=$$(openssl rand -hex 32) && [ -n "$$ONEX_MINT_PASSWORD" ] || exit 1; \
	  export ONEX_MINT_PASSWORD; \
	  run_node converge ONEX_MINT_PASSWORD || exit 1; \
	  principal=$$(jq -r '.result.handler_result.principal // empty' "$$work/receipt.json"); \
	  echo "mint-local: principal $$principal"; \
	  echo "mint-local: $$(jq -r '.result.handler_result.action' "$$work/receipt.json")"; \
	  printf '%s\n' "$$ONEX_MINT_PASSWORD" \
	    | onex auth lane-login --lane local --sasl-username "$$principal" --sasl-password-stdin || exit 1; \
	  unset ONEX_MINT_PASSWORD; \
	  probe || { echo "mint-local: the host delegation wrote no terminal event for correlation $${cid:-none}" >&2; exit 1; }; \
	  echo "mint-local: verified"

delegate-local: _check-docker ## Laptop profile: one delegation through your runtime on the local broker (PROMPT="...")
	@test -n "$(PROMPT)" || { echo 'usage: make delegate-local PROMPT="Reply with exactly one word: hello"'; exit 2; }
	docker exec -u omniinfra -e HOME=/home/omniinfra $(LOCAL_PROJECT)-runtime-effects onex delegate $(DELEGATE_FLAGS) "$(PROMPT)" \
	  --bus kafka --kafka-bootstrap redpanda:9092 --locus deployed-lane

tenant-local: local-env ## Laptop profile: give your runtime a tenant (needed to use your own provider key); then make up-local
	@if grep -q '^ONEX_TENANT_ID=.' "$(LOCAL_ENV_FILE)"; then \
	  echo "==> $(LOCAL_ENV_FILE) already names a tenant: $$(sed -n 's/^ONEX_TENANT_ID=//p' "$(LOCAL_ENV_FILE)")"; \
	else \
	  tenant="local-$$(openssl rand -hex 6)" || exit 1; \
	  if grep -q '^ONEX_TENANT_ID=' "$(LOCAL_ENV_FILE)"; then \
	    env_tmp=$$(mktemp "$(LOCAL_ENV_FILE).XXXXXX") || exit 1; \
	    sed "s|^ONEX_TENANT_ID=.*|ONEX_TENANT_ID=$$tenant|" "$(LOCAL_ENV_FILE)" > "$$env_tmp" \
	      && cat "$$env_tmp" > "$(LOCAL_ENV_FILE)"; \
	    result=$$?; rm -f "$$env_tmp"; [ "$$result" -eq 0 ] || exit "$$result"; \
	  else \
	    printf '\nONEX_TENANT_ID=%s\n' "$$tenant" >> "$(LOCAL_ENV_FILE)" || exit 1; \
	  fi; \
	  echo "==> Added ONEX_TENANT_ID=$$tenant to $(LOCAL_ENV_FILE). Run make up-local so your runtime serves it."; \
	fi

secret-local: _check-docker ## Laptop profile: register your tenant's provider key (PROVIDER=gemini; stdin or hidden prompt)
	@test -n "$(PROVIDER)" || { echo 'usage: make secret-local PROVIDER=gemini' >&2; exit 2; }
	@test -f "$(LOCAL_ENV_FILE)" || { echo 'ERROR: local env file is missing; run make tenant-local first.' >&2; exit 2; }
	@tenant=$$(sed -n 's/^ONEX_TENANT_ID=//p' "$(LOCAL_ENV_FILE)") || exit 1; \
	  test -n "$$tenant" || { echo 'ERROR: ONEX_TENANT_ID is empty or absent; run make tenant-local, then make up-local.' >&2; exit 2; }; \
	  served=$$(docker exec $(LOCAL_PROJECT)-runtime-effects printenv ONEX_TENANT_ID 2>/dev/null) || served=""; \
	  test "$$served" = "$$tenant" || { echo "ERROR: your runtime serves tenant '$$served', not '$$tenant'. Run make up-local so it serves $$tenant, then register the key." >&2; exit 2; }; \
	  if [ -t 0 ]; then \
	    bash -c 'read -r -s -p "Provider key (input hidden): " credential || exit 1; printf "\n" >&2; printf "%s" "$$credential"' \
	      | docker exec -u omniinfra -e HOME=/home/omniinfra -i $(LOCAL_PROJECT)-runtime-effects onex secret register-tenant-key "$(PROVIDER)" --tenant "$$tenant"; \
	  else \
	    docker exec -u omniinfra -e HOME=/home/omniinfra -i $(LOCAL_PROJECT)-runtime-effects onex secret register-tenant-key "$(PROVIDER)" --tenant "$$tenant"; \
	  fi

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
