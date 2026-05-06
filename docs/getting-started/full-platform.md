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

> Keycloak (auth bundle) requires secrets in `~/.omnibase/.env` and is
> covered below — see "Add Keycloak (optional)" after the **Configure
> Secrets** step. Do not run `make seed-keycloak` until that env file
> exists.

## Configure Secrets

Set up ~/.omnibase/.env with required environment variables

- [ ] **Configure Secrets**
  - Verify: `~/.omnibase/.env` (file_exists)
  - Estimated time: 2m 0s

## Add Keycloak (optional)

After completing **Configure Secrets** above, you can bring up the auth
bundle and reconcile its clients. Skip this section if you don't need
local OIDC/auth.

- [ ] **Start Keycloak**
  - Run: `make up-auth`
  - Verify: `localhost:28080/realms/omninode/.well-known/openid-configuration` (http_health)
  - Estimated time: 1m 0s
- [ ] **Reconcile Keycloak clients**
  - Run: `make seed-keycloak`
  - Prerequisite: `~/.omnibase/.env` must contain `KEYCLOAK_ADMIN_USERNAME`
    and `KEYCLOAK_ADMIN_PASSWORD` (the Makefile's `_check-env-file`
    guard catches the missing-file case with an actionable error).
  - Verify: 5 clients reconciled to `op=created` on first run, `op=unchanged`
    on subsequent runs (idempotent)
  - Estimated time: 30s

## Start Omnidash Dashboard

Start the analytics dashboard for monitoring

- [ ] **Start Omnidash Dashboard**
  - Verify: `http://localhost:3000` (http_health)
  - Estimated time: 1m 0s
