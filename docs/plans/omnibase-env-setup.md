# ~/.omnibase/.env Setup

## Overview

`~/.omnibase/.env` is the transitional home-directory env file that contains all
shared platform vars for OmniNode repos. This replaces per-repo `.env` files.

This document records the implementation for OMN-2481.

## Status

TRANSITIONAL: will shrink to 5 bootstrap lines once all docker-compose services
read from Infisical on boot (prerequisite for OMN-2484).

## Sections

| Section | Keys |
|---------|------|
| PostgreSQL | POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD |
| Kafka | KAFKA_BOOTSTRAP_SERVERS, KAFKA_ENVIRONMENT |
| Consul | CONSUL_HTTP_ADDR, CONSUL_HTTP_TOKEN |
| Vault | VAULT_ADDR, VAULT_TOKEN, VAULT_NAMESPACE |
| Infisical (bootstrap only) | INFISICAL_ADDR, INFISICAL_CLIENT_ID, INFISICAL_CLIENT_SECRET, INFISICAL_PROJECT_ID, INFISICAL_ENCRYPTION_KEY, INFISICAL_AUTH_SECRET |
| LLM | OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY, Z_AI_API_KEY, Z_AI_API_URL, LLM_CODER_URL, LLM_GENERAL_URL |
| Auth | JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRY_SECONDS, SERVICE_AUTH_TOKEN, GH_PAT |
| Valkey | VALKEY_HOST, VALKEY_PORT, VALKEY_PASSWORD |
| Service URLs | ONEX_TREE_SERVICE_URL, METADATA_STAMPING_SERVICE_URL, REMOTE_SERVER_IP |
| Environment | ENVIRONMENT, LOG_LEVEL, DEBUG |
| Slack | SLACK_BOT_TOKEN, SLACK_CHANNEL_ID |

## Rules

1. NEVER add repo-specific vars (POSTGRES_DATABASE, etc.)
2. File lives in ~ and is NEVER committed to any repo
3. Source at shell startup: add `source ~/.omnibase/.env` to `~/.zshrc`

## Shell Setup

Add to `~/.zshrc`:

```bash
# Source shared OmniNode platform vars
if [ -f ~/.omnibase/.env ]; then
    source ~/.omnibase/.env
fi
```

## Verification

After sourcing in a new terminal:

```bash
echo "POSTGRES_HOST=$POSTGRES_HOST"       # Expect: 192.168.86.200
echo "INFISICAL_ADDR=$INFISICAL_ADDR"     # Expect: http://localhost:8880
echo "LLM_CODER_URL=$LLM_CODER_URL"      # Expect: http://192.168.86.201:8000
```
