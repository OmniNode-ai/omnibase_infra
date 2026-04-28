#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
set -euo pipefail
FILE="docker/keycloak/omninode-realm.json"
if jq -e 'has("clients") or has("clientScopes")' "$FILE" > /dev/null 2>&1; then
  echo "ERROR: $FILE must not contain 'clients' or 'clientScopes'. Move to desired-clients.json." >&2
  exit 1
fi
