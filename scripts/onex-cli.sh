#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Thin CLI wrapper for the service catalog.
# Shell functions in ~/.zshrc delegate here.
# Uses uv run (matching project convention) not bare python3.

set -euo pipefail

INFRA_DIR="${OMNIBASE_INFRA_DIR:-$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")}"
ONEX_INFRA_REMOTE_LOCAL_EXIT=2

_onex_trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

_onex_env_value() {
    local key="$1"
    local env_file="${OMNIBASE_ENV_FILE:-${HOME}/.omnibase/.env}"
    local current_value=""

    if [[ -f "$env_file" ]]; then
        local line value
        line=$(grep -E "^[[:space:]]*${key}=" "$env_file" | tail -n 1 || true)
        if [[ -n "$line" ]]; then
            value="${line#*=}"
            value="$(_onex_trim "$value")"
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"
            printf '%s' "$value"
            return 0
        fi
    fi

    current_value="${!key-}"
    if [[ -n "$current_value" ]]; then
        printf '%s' "$current_value"
        return 0
    fi

    return 1
}

_onex_host_from_endpoint() {
    local endpoint="$1"
    endpoint="${endpoint%%,*}"
    endpoint="${endpoint#*://}"
    endpoint="${endpoint%%/*}"
    endpoint="${endpoint%%:*}"
    printf '%s' "$endpoint"
}

_onex_infra_target_host() {
    local postgres_host kafka_bootstrap

    postgres_host="$(_onex_env_value POSTGRES_HOST || true)"
    if [[ -n "$postgres_host" ]]; then
        _onex_host_from_endpoint "$postgres_host"
        return 0
    fi

    kafka_bootstrap="$(_onex_env_value KAFKA_BOOTSTRAP_SERVERS || true)"
    if [[ -n "$kafka_bootstrap" ]]; then
        _onex_host_from_endpoint "$kafka_bootstrap"
        return 0
    fi

    return 1
}

_onex_host_is_local() {
    local host="$1"
    case "$host" in
        "" | "localhost" | "localhost.localdomain" | "127."* | "::1" | "0.0.0.0")
            return 0
            ;;
    esac

    if [[ "$host" == "$(hostname 2>/dev/null || true)" ]]; then
        return 0
    fi

    if command -v hostname >/dev/null 2>&1; then
        if hostname -I >/dev/null 2>&1 && hostname -I | tr ' ' '\n' | grep -Fxq "$host"; then
            return 0
        fi
    fi

    if command -v ipconfig >/dev/null 2>&1; then
        local iface
        for iface in en0 en1 en2; do
            if [[ "$(ipconfig getifaddr "$iface" 2>/dev/null || true)" == "$host" ]]; then
                return 0
            fi
        done
    fi

    return 1
}

_onex_quote_words() {
    local quoted=""
    local arg
    for arg in "$@"; do
        printf -v arg '%q' "$arg"
        quoted="${quoted:+$quoted }$arg"
    done
    printf '%s' "$quoted"
}

_onex_remote_target() {
    local host="$1"
    if [[ -n "${ONEX_INFRA_REMOTE_USER:-}" ]]; then
        printf '%s@%s' "$ONEX_INFRA_REMOTE_USER" "$host"
    else
        printf '%s' "$host"
    fi
}

_onex_infra_remote_guard() {
    local entrypoint="$1"
    shift

    if [[ "${ONEX_INFRA_REMOTE_DELEGATED:-}" == "1" ]]; then
        return "$ONEX_INFRA_REMOTE_LOCAL_EXIT"
    fi

    local target_host
    target_host="$(_onex_infra_target_host || true)"
    if _onex_host_is_local "$target_host"; then
        return "$ONEX_INFRA_REMOTE_LOCAL_EXIT"
    fi

    local remote_target
    remote_target="$(_onex_remote_target "$target_host")"

    if [[ "${ONEX_INFRA_REMOTE_BEHAVIOR:-ssh}" == "fail" ]]; then
        echo "[infra] ERROR: Infrastructure is configured to run on ${target_host}." >&2
        echo "[infra] Run ${entrypoint} on that host or use: ssh ${remote_target} '${entrypoint}'" >&2
        return 1
    fi

    if ! command -v ssh >/dev/null 2>&1; then
        echo "[infra] ERROR: Infrastructure is configured to run on ${target_host}, but ssh is not available." >&2
        echo "[infra] Run ${entrypoint} on that host or use: ssh ${remote_target} '${entrypoint}'" >&2
        return 1
    fi

    local remote_args remote_cmd remote_shell
    remote_args="$(_onex_quote_words "$@")"
    remote_cmd="ONEX_INFRA_REMOTE_DELEGATED=1 ${entrypoint}${remote_args:+ $remote_args}"
    remote_shell="${ONEX_INFRA_REMOTE_SHELL:-zsh -lic}"

    echo "[infra] Infrastructure is configured for ${target_host}; delegating ${entrypoint} over SSH." >&2
    # Command is deliberately assembled and quoted locally.
    # shellcheck disable=SC2029
    ssh "$remote_target" "${remote_shell} $(_onex_quote_words "$remote_cmd")"
}

_onex_remote_guard_or_return() {
    local entrypoint="$1"
    shift
    _onex_infra_remote_guard "$entrypoint" "$@"
    local status=$?
    if [[ "$status" -eq 0 ]]; then
        return 0
    fi
    if [[ "$status" -ne "$ONEX_INFRA_REMOTE_LOCAL_EXIT" ]]; then
        return "$status"
    fi
    return "$ONEX_INFRA_REMOTE_LOCAL_EXIT"
}

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

_onex_continue_locally_or_return() {
    local status="$1"
    if [[ "$status" -ne "$ONEX_INFRA_REMOTE_LOCAL_EXIT" ]]; then
        return "$status"
    fi
    return 0
}

infra-up() {
    local status
    if _onex_remote_guard_or_return infra-up "$@"; then
        return 0
    else
        status=$?
    fi
    _onex_continue_locally_or_return "$status" || return $?
    onex_up core "$@"
}

infra-up-runtime() {
    local status
    if _onex_remote_guard_or_return infra-up-runtime "$@"; then
        return 0
    else
        status=$?
    fi
    _onex_continue_locally_or_return "$status" || return $?
    onex_up runtime "$@"
}

infra-up-memory() {
    local status
    if _onex_remote_guard_or_return infra-up-memory "$@"; then
        return 0
    else
        status=$?
    fi
    _onex_continue_locally_or_return "$status" || return $?
    onex_up runtime memgraph "$@"
}

infra-up-auth() {
    local status
    if _onex_remote_guard_or_return infra-up-auth "$@"; then
        return 0
    else
        status=$?
    fi
    _onex_continue_locally_or_return "$status" || return $?
    onex_up core auth "$@"
}

infra-down() {
    local status
    if _onex_remote_guard_or_return infra-down "$@"; then
        return 0
    else
        status=$?
    fi
    _onex_continue_locally_or_return "$status" || return $?
    onex_down "$@"
}

infra-status() {
    local status
    if _onex_remote_guard_or_return infra-status "$@"; then
        return 0
    else
        status=$?
    fi
    _onex_continue_locally_or_return "$status" || return $?
    onex_status "$@"
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
