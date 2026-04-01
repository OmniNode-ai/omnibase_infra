#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Thin CLI wrapper for the service catalog.
# Shell functions in ~/.zshrc delegate here.
# Uses uv run (matching project convention) not bare python3.

set -euo pipefail

INFRA_DIR="${OMNIBASE_INFRA_DIR:-$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")}"

onex_up() {
    local force_build=0
    local bundles=""
    for arg in "$@"; do
        if [[ "$arg" == "--build" ]]; then
            force_build=1
        else
            bundles="${bundles:+$bundles }$arg"
        fi
    done
    if [ -z "$bundles" ]; then
        bundles=$(cd "$INFRA_DIR" && uv run python -m omnibase_infra.docker.catalog.cli read-stack)
    fi

    # Auto-detect stale images when not explicitly requesting --build
    if [[ "$force_build" -eq 0 ]]; then
        local stale
        stale=$("${INFRA_DIR}/scripts/check-stale-images.sh" 2>/dev/null || true)
        if [[ -n "$stale" ]]; then
            echo "[onex] Stale images detected (code is newer than image):" >&2
            echo "$stale" | sed 's/^/  /' >&2
            echo "[onex] Auto-rebuilding..." >&2
            force_build=1
        fi
    fi

    if [[ "$force_build" -eq 1 ]]; then
        # shellcheck disable=SC2086
        (cd "$INFRA_DIR" && uv run python -m omnibase_infra.docker.catalog.cli up --build $bundles)
    else
        # shellcheck disable=SC2086
        (cd "$INFRA_DIR" && uv run python -m omnibase_infra.docker.catalog.cli up $bundles)
    fi
}

onex_down() {
    (cd "$INFRA_DIR" && uv run python -m omnibase_infra.docker.catalog.cli down)
}

onex_status() {
    (cd "$INFRA_DIR" && uv run python -m omnibase_infra.docker.catalog.cli status)
}

onex_generate() {
    local bundles="$*"
    # shellcheck disable=SC2086
    (cd "$INFRA_DIR" && uv run python -m omnibase_infra.docker.catalog.cli generate $bundles)
}

onex_validate() {
    local bundles="$*"
    # shellcheck disable=SC2086
    (cd "$INFRA_DIR" && uv run python -m omnibase_infra.docker.catalog.cli validate $bundles)
}

# Backwards-compat mapping:
#   infra-up             -> onex up core
#   infra-up-runtime     -> onex up runtime          (includes core + valkey)
#   infra-up-memory      -> onex up runtime memgraph
#   infra-up-auth        -> onex up core auth
#   infra-down           -> onex down
#   infra-status         -> onex status
#
# Intentional behavior changes:
#   infra-down-runtime: currently stops only runtime-profile containers.
#     onex down stops ALL containers in the generated compose.
#     Partial teardown is replaced by re-selecting bundles (onex up core).
#   infra-up: currently uses hardcoded compose path with worktree guards.
#     onex up uses the generated compose, eliminating worktree-path bugs.
