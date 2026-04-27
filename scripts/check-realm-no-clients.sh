#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
FILE="docker/keycloak/omninode-realm.json"
if jq -e '.clients // .clientScopes' "$FILE" > /dev/null 2>&1; then
  echo "ERROR: $FILE must not contain 'clients' or 'clientScopes'. Move to desired-clients.json." >&2
  exit 1
fi
