#!/usr/bin/env bash
# OMN-14027 C2 — snapshot and re-seed the Actions tool cache across the runner fleet.
#
# THE HAZARD THIS EXISTS FOR (found by live readback, 2026-08-14)
# `RUNNER_TOOL_CACHE` for this fleet is /home/runner/actions-runner/_work/_tool,
# which lives in the CONTAINER FILESYSTEM. Only /home/runner/.runner-creds is a
# named volume. So `docker compose up --force-recreate` on the runner fleet
# DESTROYS the entire fleet-wide tool cache. The measured steady state on
# 2026-08-14 was 72/72 runners warm (Python 3.12.x + 3.13.15, uv 0.6.x-0.12.x);
# a naive fleet recreate would take that to 0/72 and hand the next wave 72
# simultaneous cold CPython + uv downloads from codeload/objects.githubusercontent
# -- precisely the egress stampede OMN-14027 exists to remove.
#
# Any runner-fleet recreate MUST therefore be bracketed:
#     toolcache-seed.sh snapshot            # before  (donor -> host snapshot)
#     <recreate the runners>
#     toolcache-seed.sh restore             # after   (host snapshot -> containers)
#
# `restore` is also the way to normalise version drift: runners created at
# different times hold different CPython patch releases (measured 2026-08-14:
# 3.12.14 on runners 1 + 65-72, 3.12.13 on 2-64). A `python-version: "3.12"`
# request is satisfied by either, so that drift is currently benign -- but an
# exact pin or `check-latest: true` turns it into a 63-way cold download.
#
# Runs on the runner host as the `jonah` user (needs docker access, not root).

set -euo pipefail

SNAPSHOT_ROOT="${OMNI_TOOLCACHE_SNAPSHOT_ROOT:-${HOME}/.omnibase/runners/toolcache-seed}"
CONTAINER_TOOL_CACHE="/home/runner/actions-runner/_work/_tool"
RUNNER_FILTER="${OMNI_RUNNER_FILTER:-^omninode-runner-}"

usage() {
    cat >&2 <<'EOF'
usage: toolcache-seed.sh <snapshot|restore|report> [container ...]

  snapshot   Merge every runner's tool cache into the host snapshot at
             $OMNI_TOOLCACHE_SNAPSHOT_ROOT. Union, never destructive: a version
             present on any runner ends up in the snapshot.
  restore    Copy every snapshot entry that a container is MISSING into that
             container. Never overwrites an existing entry, so it is safe to
             re-run and cannot corrupt a live cache.
  report     Print the per-runner tool-cache inventory (the drift readback).

With no container arguments, all containers matching $OMNI_RUNNER_FILTER are used.
EOF
    exit 2
}

runner_containers() {
    if [[ $# -gt 0 ]]; then
        printf '%s\n' "$@"
    else
        docker ps --format '{{.Names}}' | grep -E "${RUNNER_FILTER}" | sort -t- -k3 -n
    fi
}

# tool/version pairs present in a container, e.g. "Python/3.12.13"
container_entries() {
    docker exec -u runner "$1" bash -lc \
        "cd ${CONTAINER_TOOL_CACHE} 2>/dev/null && find . -maxdepth 2 -mindepth 2 -type d | sed 's|^\./||'" 2>/dev/null || true
}

cmd_report() {
    local c entries
    while read -r c; do
        [[ -n "${c}" ]] || continue
        entries="$(container_entries "${c}" | sort | tr '\n' ' ')"
        printf '%s\t%s\n' "${c}" "${entries}"
    done < <(runner_containers "$@")
}

cmd_snapshot() {
    mkdir -p "${SNAPSHOT_ROOT}"
    local c entry tool version
    while read -r c; do
        [[ -n "${c}" ]] || continue
        while read -r entry; do
            [[ -n "${entry}" ]] || continue
            tool="${entry%%/*}"
            version="${entry##*/}"
            if [[ -d "${SNAPSHOT_ROOT}/${tool}/${version}" ]]; then
                continue
            fi
            echo "[toolcache-seed] snapshot ${tool}/${version} from ${c}"
            mkdir -p "${SNAPSHOT_ROOT}/${tool}"
            # Copy the version dir plus its sibling <arch>.complete markers --
            # setup-python/setup-uv treat a version without the marker as absent.
            docker cp "${c}:${CONTAINER_TOOL_CACHE}/${tool}/${version}" \
                "${SNAPSHOT_ROOT}/${tool}/${version}.partial" >/dev/null
            mv "${SNAPSHOT_ROOT}/${tool}/${version}.partial" "${SNAPSHOT_ROOT}/${tool}/${version}"
        done < <(container_entries "${c}")
    done < <(runner_containers "$@")
    echo "[toolcache-seed] snapshot at ${SNAPSHOT_ROOT}:"
    du -sh "${SNAPSHOT_ROOT}"/* 2>/dev/null || true
}

cmd_restore() {
    [[ -d "${SNAPSHOT_ROOT}" ]] || { echo "[toolcache-seed] no snapshot at ${SNAPSHOT_ROOT}; run 'snapshot' first." >&2; exit 1; }
    local c have tool_path tool version
    while read -r c; do
        [[ -n "${c}" ]] || continue
        have="$(container_entries "${c}" | sort | tr '\n' ' ')"
        docker exec -u runner "${c}" mkdir -p "${CONTAINER_TOOL_CACHE}" >/dev/null 2>&1 || true
        for tool_path in "${SNAPSHOT_ROOT}"/*/*; do
            [[ -d "${tool_path}" ]] || continue
            version="$(basename "${tool_path}")"
            tool="$(basename "$(dirname "${tool_path}")")"
            case " ${have} " in
                *" ${tool}/${version} "*) continue ;;
            esac
            echo "[toolcache-seed] restore ${tool}/${version} -> ${c}"
            docker exec -u runner "${c}" mkdir -p "${CONTAINER_TOOL_CACHE}/${tool}" >/dev/null
            docker cp "${tool_path}" "${c}:${CONTAINER_TOOL_CACHE}/${tool}/" >/dev/null
            # docker cp lands as root; the runner user must own its own cache.
            docker exec -u root "${c}" chown -R runner:runner "${CONTAINER_TOOL_CACHE}/${tool}/${version}" >/dev/null
            # The `<arch>.complete` marker that setup-python/setup-uv key on
            # lives INSIDE the version directory (verified live: the tree is
            # `Python/3.12.14/x64` alongside `Python/3.12.14/x64.complete`), so
            # copying the version directory carries the marker with it. Assert
            # that rather than assume it -- a version without its marker is
            # invisible to the actions and would silently re-download.
            if ! docker exec -u runner "${c}" bash -lc \
                "ls ${CONTAINER_TOOL_CACHE}/${tool}/${version}/*.complete >/dev/null 2>&1"; then
                echo "[toolcache-seed] WARNING: ${tool}/${version} on ${c} has no .complete marker; the action will re-download it." >&2
            fi
        done
    done < <(runner_containers "$@")
}

[[ $# -ge 1 ]] || usage
action="$1"
shift || true
case "${action}" in
    snapshot) cmd_snapshot "$@" ;;
    restore) cmd_restore "$@" ;;
    report) cmd_report "$@" ;;
    *) usage ;;
esac
