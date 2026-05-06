<!-- GENERATED FROM canonical.yaml -- DO NOT EDIT MANUALLY -->

# Full Platform Setup

## Check Python Installation

Verify Python 3.12+ is available

- [ ] **Check Python Installation**
  - Verify: `python3 --version` (command_exit_0)
  - Estimated time: 5s

## Install uv Package Manager

Install uv for dependency management

- [ ] **Install uv Package Manager**
  - Verify: `uv --version` (command_exit_0)
  - Estimated time: 30s

## Install omnibase_core

Install the core ONEX package with uv

- [ ] **Install omnibase_core**
  - Verify: `omnibase_core` (python_import)
  - Estimated time: 1m 0s

## Start Docker Infrastructure

Start PostgreSQL, Redpanda, Valkey, and Infisical via the in-repo Makefile.
The `Makefile` (OMN-10377) detects a missing/stopped Docker daemon and emits
an actionable error before doing anything destructive.

- [ ] **Start Docker Infrastructure**
  - Run: `make up`
  - Verify: `localhost:5436` (tcp_probe) and `make status` shows all
    `omnibase-infra-*` containers as healthy
  - Estimated time: 2m 0s

To add Keycloak for local OIDC/auth flows, follow up with `make up-auth` and
then `make seed-keycloak` (requires `~/.omnibase/.env` with
`KEYCLOAK_ADMIN_USERNAME` and `KEYCLOAK_ADMIN_PASSWORD`).

## Configure Secrets

Set up ~/.omnibase/.env with required environment variables

- [ ] **Configure Secrets**
  - Verify: `~/.omnibase/.env` (file_exists)
  - Estimated time: 2m 0s

## Start Omnidash Dashboard

Start the analytics dashboard for monitoring

- [ ] **Start Omnidash Dashboard**
  - Verify: `http://localhost:3000` (http_health)
  - Estimated time: 1m 0s
