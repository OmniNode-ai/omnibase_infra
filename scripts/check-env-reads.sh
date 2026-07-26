#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# CI gate: block new os.environ/os.getenv reads outside approved modules,
# AND block reads of undeclared secret-ish names (OMN-14951 gap 2).
# Modes:
#   --staged  (pre-commit): check staged diff only
#   --base <ref>  (CI): check diff against base branch
set -euo pipefail

MODE="${1:---staged}"
BASE_REF="${2:-origin/main}"

# Anchored path patterns — matched against /-separated path segments to prevent
# substring false-positives (e.g. "tests/" must not match "src/contests/").
# Prefix patterns: file must START with this value (top-level directories).
APPROVED_PREFIX_PATTERNS=(
    "tests/"
    "scripts/"
)
# Infix patterns: file must contain "/<pattern>" (path segment boundary).
APPROVED_INFIX_PATTERNS=(
    "/runtime/service_kernel.py"
    "/runtime/overlay/"
    "/runtime/config_discovery/config_prefetcher.py"
    "/runtime/runtime_profile.py"
    "/services/registry_api/registry_discovery.py"
    # OMN-13537: receipt-mode CLI is the config-resolution boundary between the
    # --state-root flag and the ArtifactStore's sole env-var contract
    # (ONEX_ARTIFACT_STORE_ROOT). The pinned omnibase_core store exposes no
    # injection seam, so the boundary must publish the resolved default to the
    # environment — same class of boundary as service_kernel/config_prefetcher.
    "/cli/receipt_mode.py"
    # OMN-14208: the state_io opt-in dispatch seam resolves its REQUIRED DSN
    # (per-service *_DB_URL, keyed via _DB_URL_ENV_MAP) at wiring time — the
    # exact same env-var-DSN-resolution boundary the pre-existing (pre-gate)
    # db_io projection path in this same file already reads (_DB_URL_ENV_MAP
    # / os.environ.get(db_url_env, "")). Formalizing the allowlist entry
    # rather than duplicating that read through a new indirection layer.
    "/runtime/auto_wiring/handler_wiring.py"
    # OMN-14208: state_io's stale-in-flight-row TTL (DELEGATION_STALE_TTL_SECONDS)
    # is a narrowly-scoped, single-purpose module dedicated to this seam.
    "/runtime/state_io/"
    # OMN-14597: model_postgres_pool_config.py's from_env() already reads
    # POSTGRES_POOL_MIN_SIZE/MAX_SIZE via os.getenv() pre-gate, in the exact
    # same method. POSTGRES_SSL_MODE/POSTGRES_SSL_CA_FILE (needed so the
    # runtime can reach RDS, which enforces TLS) are the same class of
    # env-var read for the same factory, not a new pattern introduced
    # elsewhere.
    "/runtime/models/model_postgres_pool_config.py"
)

ENV_READ_PATTERNS='os\.environ\[|os\.environ\.get|os\.getenv|from os import environ|from os import getenv'

is_approved() {
    local file="$1"
    for prefix in "${APPROVED_PREFIX_PATTERNS[@]}"; do
        [[ "$file" == "$prefix"* ]] && return 0
    done
    for infix in "${APPROVED_INFIX_PATTERNS[@]}"; do
        [[ "$file" == *"$infix"* ]] && return 0
    done
    return 1
}

# OMN-14951 gap 2: files subject to the secret-name declaration check are
# exactly the boundary infix set above -- these ARE a deployable's actual
# env-consuming boundary code (OMN-14951 scope: "every deployable's consumed
# env is enumerated and declared"). tests/ and scripts/ stay exempt from THIS
# check: test doubles and tooling routinely mock/set secret-ish env names and
# are not a deployable's consumed env surface. (They remain fully exempt from
# the path-boundary check above, unchanged.)
is_secret_name_check_target() {
    local file="$1"
    for infix in "${APPROVED_INFIX_PATTERNS[@]}"; do
        [[ "$file" == *"$infix"* ]] && return 0
    done
    return 1
}

get_changed_files() {
    case "$MODE" in
        --staged) git diff --cached --diff-filter=ACMR --name-only -- '*.py' ;;
        --base)   git diff "$BASE_REF"...HEAD --diff-filter=ACMR --name-only -- '*.py' ;;
        *)        echo "Unknown mode: $MODE" >&2; exit 1 ;;
    esac
}

get_diff() {
    local file="$1"
    case "$MODE" in
        --staged) git diff --cached -- "$file" ;;
        --base)   git diff "$BASE_REF"...HEAD -- "$file" ;;
    esac
}

# OMN-14951 gap 2: statically scan os.environ[...]/os.environ.get(...)/
# os.getenv(...)/get_secret(...) reads of secret-ish names in boundary files
# and fail when the name is neither the finite, named bootstrap allowlist nor
# self-declared as a required_secrets/bootstrap_secrets list element in the
# same file. Embedded as a single python3 invocation (stdlib only, no new
# dependency) rather than a second parallel scanner script — this stays one
# committed gate file, converging with the path-boundary check above.
check_secret_name_declarations() {
    local target_files=()
    while IFS= read -r file; do
        [ -z "$file" ] && continue
        is_secret_name_check_target "$file" || continue
        [ -f "$file" ] || continue  # deleted files: nothing to scan
        target_files+=("$file")
    done < <(get_changed_files)

    [ ${#target_files[@]} -eq 0 ] && return 0

    local tmp_data tmp_py rc
    tmp_data="$(mktemp)"
    tmp_py="$(mktemp)"

    {
        for f in "${target_files[@]}"; do
            printf '##FILE##%s\n' "$f"
            get_diff "$f"
            printf '\n'
        done
    } > "$tmp_data"

    cat > "$tmp_py" <<'PYEOF'
import re
import sys

# Finite, named bootstrap allowlist (OMN-14951): the irreducible set of
# secret-ish-looking env vars needed to reach Infisical / identify the
# runtime lane / locate the workspace in the first place — "a keyring
# cannot unlock itself". Keep this list explicit and small; do NOT widen it
# to cover a real declared-required secret (those must route through
# ModelSecretResolverConfig.required_secrets instead, per the class-level
# gate this script's sibling check enforces at construction/resolution
# time).
BOOTSTRAP_ALLOWLIST = {
    "INFISICAL_ADDR",
    "INFISICAL_CLIENT_ID",
    "INFISICAL_CLIENT_SECRET",
    "INFISICAL_PROJECT_ID",
    "INFISICAL_ENVIRONMENT",
    "RUNTIME_PROFILE",
    "OMNI_HOME",
    "ONEX_SECRET_POLICY",
    "ONEX_SECRET_RESOLVER_CONFIG_PATH",
}

SECRET_ISH = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|CERT|PRIVATE|DSN)",
    re.IGNORECASE,
)
READ_PATTERN = re.compile(
    r"""os\.environ\[\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*\]
        |os\.environ\.get\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']
        |os\.getenv\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']
        |get_secret(?:_async)?\(\s*["']([A-Za-z_][A-Za-z0-9_.]*)["']
    """,
    re.VERBOSE,
)
DECLARATION_MARKER = re.compile(r"required_secrets|bootstrap_secrets")


def main() -> int:
    data_path = sys.argv[1]
    with open(data_path, encoding="utf-8") as fh:
        raw = fh.read()

    failed = False
    for section in raw.split("##FILE##"):
        if not section.strip():
            continue
        file_line, _, diff_body = section.partition("\n")
        file = file_line.strip()
        if not file:
            continue

        try:
            with open(file, encoding="utf-8") as fh:
                full_content = fh.read()
        except OSError:
            full_content = ""
        declares_lists = bool(DECLARATION_MARKER.search(full_content))

        for line in diff_body.splitlines():
            if not line.startswith("+") or line.startswith("+++"):
                continue
            for match in READ_PATTERN.finditer(line):
                name = next(g for g in match.groups() if g)
                if not SECRET_ISH.search(name):
                    continue
                if name in BOOTSTRAP_ALLOWLIST:
                    continue
                if declares_lists and re.search(
                    r"""["']""" + re.escape(name) + r"""["']""", full_content
                ):
                    continue
                print(
                    f"BLOCKED: {file} reads undeclared secret-ish name '{name}'"
                )
                print(
                    "  Declare it in required_secrets/bootstrap_secrets "
                    "(OMN-14951) or the bootstrap allowlist."
                )
                failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
PYEOF

    python3 "$tmp_py" "$tmp_data"
    rc=$?
    rm -f "$tmp_data" "$tmp_py"
    return $rc
}

FAILED=0
while IFS= read -r file; do
    [ -z "$file" ] && continue
    is_approved "$file" && continue

    if get_diff "$file" | grep -qE "^\+.*($ENV_READ_PATTERNS)"; then
        echo "BLOCKED: $file introduces new os.environ/os.getenv read"
        echo "  Use overlay-resolved config instead."
        echo "  Approved top-level dirs: ${APPROVED_PREFIX_PATTERNS[*]}"
        echo "  Approved path segments: ${APPROVED_INFIX_PATTERNS[*]}"
        FAILED=1
    fi
done < <(get_changed_files)

if ! check_secret_name_declarations; then
    FAILED=1
fi

exit $FAILED
