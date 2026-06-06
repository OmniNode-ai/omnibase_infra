#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# sync-node-migrations.sh — Auto-discover omnimarket node-owned migrations and
# vendor them into omnibase_infra's namespaced forward-migration location.
#
# Ticket: OMN-12559
#
# WHY THIS EXISTS
#   omnimarket projection nodes ship SQL under
#     src/omnimarket/nodes/<node>/migrations/*.sql
#   The omnibase_infra forward-migration runner only reads
#     docker/migrations/forward/. Previously, landing a node-owned view in infra
#   required a manual host copy AND a manual renumber (e.g. savings 076 -> 084)
#   to dodge a numeric collision with the flat infra sequence. That renumber is
#   an operational footgun: the view in the source repo and the file in infra
#   drift, and every new node migration repeats the dance.
#
# WHAT IT DOES
#   Discovers every src/omnimarket/nodes/<node>/migrations/*.sql in the resolved
#   omnimarket source tree and mirrors it 1:1 into
#     docker/migrations/forward/nodes/<node>/<file>.sql
#   The forward-migration runner (run-forward-migrations.sh) applies these under
#   a NAMESPACED migration_id  node:<node>:<file>  which lives in a separate
#   identity space from the flat infra sequence — so NO renumber is ever needed.
#
#   The vendored files are committed to omnibase_infra so a clean clone
#   reproduces the views without any manual copy. Re-run this script (and commit
#   the diff) whenever omnimarket adds or changes a node migration.
#
# RESOLUTION ORDER for the omnimarket source tree:
#   1. $OMNIMARKET_SRC                     (explicit override; repo root)
#   2. $OMNI_HOME/omnimarket               (canonical clone registry)
#   3. installed package: python -c 'import omnimarket; print(parent)'
#
# NODE SELECTION
#   Discovery is scoped by the allowlist (one entry per line) in
#     docker/migrations/forward/nodes/.synced-nodes
#   Each entry is either:
#     <node>                 vendor ALL of that node's migrations, or
#     <node>:<glob>          vendor only files whose basename matches <glob>
#                            (shell case-glob, e.g. 076_*.sql)
#   This keeps the vendored tree deterministic: only nodes whose projection
#   tables/views omnibase_infra is responsible for materializing are pulled in,
#   and --check stays consistent with what is committed. Add an entry and re-run
#   to onboard a new node's migrations.
#
# USAGE
#   scripts/sync-node-migrations.sh            # vendor (writes files)
#   scripts/sync-node-migrations.sh --check    # CI mode: fail if drift exists
#
# EXIT CODES
#   0 — in sync (or vendored successfully)
#   1 — --check mode and vendored tree differs from source (drift)
#   2 — could not resolve omnimarket source tree

set -euo pipefail

CHECK_MODE=0
if [ "${1:-}" = "--check" ]; then
  CHECK_MODE=1
fi

# Repo root = parent of this script's directory.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_ROOT="${REPO_ROOT}/docker/migrations/forward/nodes"
SYNCED_NODES_FILE="${DEST_ROOT}/.synced-nodes"

# Load the node allowlist (one node name per line; '#' comments and blanks ok).
load_synced_nodes() {
  if [ ! -f "${SYNCED_NODES_FILE}" ]; then
    echo "[sync-node-migrations] ERROR: missing node allowlist ${SYNCED_NODES_FILE}" >&2
    exit 2
  fi
  grep -vE '^\s*(#|$)' "${SYNCED_NODES_FILE}" | tr -d ' \t'
}

# Return 0 if (node, filename) is selected by the allowlist.
# Entries are  <node>  (all files) or  <node>:<glob>  (basename glob match).
file_is_allowed() {
  _node="$1"
  _file="$2"
  for _entry in ${SYNCED_NODES}; do
    _entry_node="${_entry%%:*}"
    if [ "${_entry_node}" != "${_node}" ]; then
      continue
    fi
    if [ "${_entry}" = "${_entry_node}" ]; then
      # Bare node entry: all files allowed.
      return 0
    fi
    _glob="${_entry#*:}"
    # shellcheck disable=SC2254
    case "${_file}" in
      ${_glob}) return 0 ;;
    esac
  done
  return 1
}

resolve_omnimarket_src() {
  if [ -n "${OMNIMARKET_SRC:-}" ] && [ -d "${OMNIMARKET_SRC}/src/omnimarket/nodes" ]; then
    echo "${OMNIMARKET_SRC}"
    return 0
  fi
  if [ -n "${OMNI_HOME:-}" ] && [ -d "${OMNI_HOME}/omnimarket/src/omnimarket/nodes" ]; then
    echo "${OMNI_HOME}/omnimarket"
    return 0
  fi
  # Installed package: locate the directory that contains the 'nodes' package.
  pkg_nodes="$(python3 - <<'PY' 2>/dev/null || true
import importlib.util
import pathlib

spec = importlib.util.find_spec("omnimarket")
if spec and spec.submodule_search_locations:
    base = pathlib.Path(list(spec.submodule_search_locations)[0])
    nodes = base / "nodes"
    if nodes.is_dir():
        # Echo the repo-root-equivalent: parent of src/omnimarket -> stop at base.
        print(base)
PY
)"
  if [ -n "${pkg_nodes}" ] && [ -d "${pkg_nodes}/nodes" ]; then
    # For an installed package, point NODES_DIR directly (no src/ prefix).
    echo "PKG:${pkg_nodes}"
    return 0
  fi
  return 1
}

OMK_RESOLVED="$(resolve_omnimarket_src || true)"
if [ -z "${OMK_RESOLVED}" ]; then
  # In --check mode (pre-commit / CI), the omnimarket source tree is frequently
  # not checked out. The vendored files are the durable source of truth and are
  # validated independently by the migration tests; a drift check is only
  # meaningful when the source is present. Skip (success) rather than fail so the
  # gate never produces false negatives on runners without omnimarket.
  if [ "${CHECK_MODE}" -eq 1 ]; then
    echo "[sync-node-migrations] omnimarket source not resolvable — skipping drift check." >&2
    exit 0
  fi
  echo "[sync-node-migrations] ERROR: could not resolve omnimarket source tree." >&2
  echo "  Set OMNIMARKET_SRC=<omnimarket repo root>, or OMNI_HOME=<omni_home>, or pip install omnimarket." >&2
  exit 2
fi

case "${OMK_RESOLVED}" in
  PKG:*)
    NODES_DIR="${OMK_RESOLVED#PKG:}/nodes"
    ;;
  *)
    NODES_DIR="${OMK_RESOLVED}/src/omnimarket/nodes"
    ;;
esac

SYNCED_NODES="$(load_synced_nodes)"

echo "[sync-node-migrations] omnimarket nodes: ${NODES_DIR}"
echo "[sync-node-migrations] vendoring into:   ${DEST_ROOT}"
echo "[sync-node-migrations] allowlisted nodes: ${SYNCED_NODES}"

DRIFT=0
COPIED=0

# Discover every node migration file and mirror it (allowlisted nodes only).
while IFS= read -r src_file; do
  # src_file = ${NODES_DIR}/<node>/migrations/<file>.sql
  node_name="$(basename "$(dirname "$(dirname "${src_file}")")")"
  filename="$(basename "${src_file}")"
  if ! file_is_allowed "${node_name}" "${filename}"; then
    continue
  fi
  dest_dir="${DEST_ROOT}/${node_name}"
  dest_file="${dest_dir}/${filename}"

  if [ "${CHECK_MODE}" -eq 1 ]; then
    if [ ! -f "${dest_file}" ] || ! cmp -s "${src_file}" "${dest_file}"; then
      echo "[sync-node-migrations] DRIFT: ${node_name}/${filename}" >&2
      DRIFT=1
    fi
  else
    mkdir -p "${dest_dir}"
    if [ ! -f "${dest_file}" ] || ! cmp -s "${src_file}" "${dest_file}"; then
      cp "${src_file}" "${dest_file}"
      echo "[sync-node-migrations]   vendored ${node_name}/${filename}"
      COPIED=$((COPIED + 1))
    fi
  fi
done < <(find "${NODES_DIR}" -type f -path "*/migrations/*.sql" | sort)

if [ "${CHECK_MODE}" -eq 1 ]; then
  if [ "${DRIFT}" -eq 1 ]; then
    echo "[sync-node-migrations] node migration vendor tree is OUT OF SYNC with omnimarket." >&2
    echo "  Run: scripts/sync-node-migrations.sh  then commit the diff." >&2
    exit 1
  fi
  echo "[sync-node-migrations] check: in sync."
  exit 0
fi

echo "[sync-node-migrations] done: ${COPIED} file(s) updated."
