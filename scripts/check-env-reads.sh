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
    # OMN-15234: the OMN-15233 entry for
    # /nodes/node_runner_fleet_health_compute/handlers/handler_runner_fleet_health_evaluate.py
    # was REMOVED here. It existed only because the old matcher could not tell
    # "edit the default of a pre-existing read" from "introduce a new read";
    # the per-name grandfathering below handles that case on its own, so the
    # allowlist no longer has to be widened to maintain an existing read.
    # Rule 10: narrow the matcher, do not allowlist past it.
    # OMN-16106: the evidence-autoclose sweep dispatches via RuntimeLocal's
    # single-shot compute path (_run_compute — no terminal_event / event
    # bus), which only auto-injects state_root/event_bus into a handler's
    # constructor (see RuntimeLocal._instantiate_handler). There is no
    # config-prefetch/overlay seam reachable from that path for a node's own
    # secret (LINEAR_API_KEY) or its kill switch (ONEX_AUTOCLOSE_DISABLED) —
    # same class of "no injection seam, so the boundary reads env" as the
    # /cli/receipt_mode.py entry above. Declares required_secrets per the
    # OMN-14951 gap-2 check below.
    "/nodes/node_evidence_autoclose_sweep_effect/handlers/handler_evidence_autoclose_sweep.py"
)

ENV_READ_PATTERNS='os\.environ\[|os\.environ\.get|os\.getenv|from os import environ|from os import getenv'

# OMN-15234: name-extracting form of the patterns above. Only the literal
# forms carry an env-var name; `from os import environ` and any dynamic read
# (`os.environ.get(name_var)`) deliberately do NOT match, so they fall through
# to the fail-closed branch instead of being silently grandfathered.
_Q='["'"'"']'
ENV_NAME_READ_PATTERN="(os\.environ\[|os\.environ\.get\(|os\.getenv\()[[:space:]]*${_Q}[A-Za-z_][A-Za-z0-9_]*${_Q}"

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

# OMN-15234: the commit the diff above is computed against. This is what
# "already read in this file before the change" is resolved from, so it MUST
# match get_diff's endpoints exactly:
#   --staged -> HEAD (the commit the index sits on top of)
#   --base   -> merge-base(BASE_REF, HEAD), because get_diff uses the
#               three-dot `BASE_REF...HEAD` form.
# Empty (unborn HEAD, unfetched base) means "no base to grandfather from" and
# every added read is treated as new -- fail closed.
BASE_COMMIT=""
BASE_LABEL=""
resolve_base_commit() {
    case "$MODE" in
        --staged)
            BASE_COMMIT="$(git rev-parse --verify --quiet HEAD || true)"
            BASE_LABEL="HEAD"
            ;;
        --base)
            BASE_COMMIT="$(git merge-base "$BASE_REF" HEAD 2>/dev/null || true)"
            BASE_LABEL="merge-base($BASE_REF, HEAD)"
            ;;
    esac
}

# OMN-15234: env-var names read from a blob of Python source on stdin.
# Deliberately literal-only -- see ENV_NAME_READ_PATTERN.
#
# OMN-16267: `grep -oE` matches within a single line -- it cannot see a
# read whose opening `os.getenv(`/`os.environ.get(`/`os.environ[` is on one
# line and whose quoted name is on the next (a common ruff-format output
# shape for a call with a long default expression, e.g.
#     x = os.getenv(
#         "SOME_NAME", "default"
#     )
# ). Un-caught, this silently drops such reads from `base_names`, so
# grandfathering (OMN-15234's "default/whitespace/format edit on an
# existing read -> ALLOW" rule) never fires for them -- editing ONLY the
# default value of a pre-existing multi-line-formatted read was
# indistinguishable from introducing a brand new one. `tr '\n' ' '` joins
# continuation lines into the same "line" before matching, which is
# sufficient because the pattern already tolerates arbitrary whitespace
# (`[[:space:]]*`) between the call-open token and the quote -- it does
# not depend on GNU-only `grep -P`/`-z`, so it stays portable across the
# BSD grep on this Mac and the GNU grep in CI/.200.
extract_env_names() {
    tr '\n' ' ' \
        | grep -oE "$ENV_NAME_READ_PATTERN" \
        | grep -oE "${_Q}[A-Za-z_][A-Za-z0-9_]*${_Q}" \
        | tr -d "\"'" \
        | sort -u
}

report_block() {
    local file="$1" reason="$2"
    echo "BLOCKED: $file introduces new os.environ/os.getenv read -- $reason"
    echo "  Use overlay-resolved config instead."
    echo "  Approved top-level dirs: ${APPROVED_PREFIX_PATTERNS[*]}"
    echo "  Approved path segments: ${APPROVED_INFIX_PATTERNS[*]}"
}

# OMN-15234: the policy this gate enforces is "no NEW env read", but the
# pre-OMN-15234 matcher fired on ANY added line containing os.environ. Because
# a one-character edit to a read's default re-adds that whole line, editing a
# pre-existing, already-grandfathered read was indistinguishable from
# introducing a new one -- there is no formulation of "change 900 to 4500"
# that avoids re-adding the line. That made every grandfathered read's default
# permanently unmaintainable outside the allowlist, which is how the OMN-15233
# allowlist entry (removed above) got added.
#
# Narrowed rule: an added read is allowed ONLY when the SAME env-var name is
# already read in the SAME file's base version. Consequences, all intended:
#   - new name in an existing file           -> BLOCK (name absent from base)
#   - new file (base blob does not exist)    -> BLOCK
#   - read moved to a DIFFERENT file         -> BLOCK (that file's base has no
#                                               such read; the boundary moved)
#   - read relocated within the same file    -> ALLOW (same file, same name)
#   - default/whitespace/format edit on an
#     existing read                          -> ALLOW  <- the OMN-15234 case
#   - read deleted                           -> ALLOW (no added read at all)
#   - non-literal or unnamed read form       -> BLOCK (nothing to match on)
check_new_env_reads() {
    local file="$1"
    local added_reads base_names names name blocked=0

    added_reads="$(get_diff "$file" \
        | grep -E "^\+" \
        | grep -v "^+++" \
        | grep -E "($ENV_READ_PATTERNS)" || true)"
    [ -z "$added_reads" ] && return 0

    base_names=""
    if [ -n "$BASE_COMMIT" ]; then
        base_names="$(git show "$BASE_COMMIT:$file" 2>/dev/null | extract_env_names || true)"
    fi

    while IFS= read -r line; do
        [ -z "$line" ] && continue
        names="$(printf '%s\n' "$line" | extract_env_names || true)"
        if [ -z "$names" ]; then
            report_block "$file" \
                "added read has no literal env-var name to match against $BASE_LABEL (dynamic name, or a bare 'from os import environ/getenv' import)"
            blocked=1
            continue
        fi
        while IFS= read -r name; do
            [ -z "$name" ] && continue
            if printf '%s\n' "$base_names" | grep -qx -- "$name"; then
                # Grandfathered: this exact name is already read in this exact
                # file at the base commit. Editing it is maintenance, not a new
                # env-resolution surface.
                continue
            fi
            report_block "$file" \
                "env-var name '$name' is not read in this file at $BASE_LABEL"
            blocked=1
        done <<EOF
$names
EOF
    done <<EOF
$added_reads
EOF

    return $blocked
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

resolve_base_commit

FAILED=0
while IFS= read -r file; do
    [ -z "$file" ] && continue
    is_approved "$file" && continue

    if ! check_new_env_reads "$file"; then
        FAILED=1
    fi
done < <(get_changed_files)

if ! check_secret_name_declarations; then
    FAILED=1
fi

exit $FAILED
