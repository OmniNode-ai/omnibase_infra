#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
#
# ONEX Infrastructure Runtime Entrypoint
#
# This entrypoint stamps schema fingerprints into db_metadata for ALL
# databases with schema manifests before starting the runtime kernel.
# The fingerprint is computed from the live database schema via the
# installed util_schema_fingerprint module.
# Without the fingerprint stamp, the kernel's startup assertion finds
# expected_schema_fingerprint = NULL and crash-loops.
#
# Environment:
#   OMNIBASE_INFRA_DB_URL    (required) - PostgreSQL DSN for the infra database
#   OMNIINTELLIGENCE_DB_URL  (optional) - PostgreSQL DSN for the intelligence database
#
# Usage (called automatically by Docker ENTRYPOINT):
#   entrypoint-runtime.sh <CMD args...>
#
# The script exec's into "$@" (the CMD) so the kernel process replaces
# the shell and receives signals directly from tini.

set -e

# =============================================================================
# Fresh Volume Bootstrap
# =============================================================================
# Docker named volumes mounted at /app/data or /app/logs hide the image-owned
# directories created during build. Fresh named volumes are commonly root-owned,
# so the non-root runtime user cannot write Bifrost contracts or runtime state
# unless ownership is repaired before dropping privileges.

if [ "$(id -u)" -eq 0 ]; then
  echo "[entrypoint] Bootstrapping runtime volume ownership..."
  # A folded effects role retains its original local delegation claims and
  # archives. Fresh named volumes need the same ownership bootstrap as data.
  # Only the volume, state root and existing delegation directory need ownership.
  # Preserved claim files and archives retain their existing runtime ownership.
  if [ -n "${ONEX_FOLDED_EFFECTS_STATE_ROOT:-}" ]; then
    case "$ONEX_FOLDED_EFFECTS_STATE_ROOT" in
      # Traversal is refused by segment, not substring: a name such as
      # state..v2 is one ordinary directory under the volume.
      */ | /app/effects-data/. | /app/effects-data/.. | /app/effects-data/*/*) folded_root_ok=0 ;;
      /app/effects-data/?*) folded_root_ok=1 ;;
      *) folded_root_ok=0 ;;
    esac
    if [ "$folded_root_ok" -ne 1 ]; then
      echo "[entrypoint] FOLDED_EFFECTS_STATE_REFUSED: ONEX_FOLDED_EFFECTS_STATE_ROOT must name one directory directly under /app/effects-data" >&2
      exit 2
    fi
    # Validate the existing tree before any root-owned filesystem operation.
    # install -d follows links even when chown is later told not to follow them.
    # Every component under the volume is checked: the root is one segment deep.
    folded_paths_ok() {
      for folded_path in /app/effects-data "$ONEX_FOLDED_EFFECTS_STATE_ROOT" "$ONEX_FOLDED_EFFECTS_STATE_ROOT/delegation"; do
        if [ -L "$folded_path" ]; then
          echo "[entrypoint] FOLDED_EFFECTS_STATE_REFUSED: folded state path must not be a symlink: $folded_path" >&2
          return 1
        fi
        if [ -e "$folded_path" ] && [ ! -d "$folded_path" ]; then
          echo "[entrypoint] FOLDED_EFFECTS_STATE_REFUSED: folded state path must be a directory: $folded_path" >&2
          return 1
        fi
      done
    }
    folded_paths_ok || exit 2
  fi
  install -d -o omniinfra -g omniinfra /app/data /app/data/delegation /app/logs /app/tmp
  chown -R omniinfra:omniinfra /app/data /app/logs /app/tmp
  if [ -n "${ONEX_FOLDED_EFFECTS_STATE_ROOT:-}" ]; then
    # Pin each directory inode with O_NOFOLLOW before changing its ownership.
    # No recursive walk or pathname-based chown runs over preserved contents.
    python - "$ONEX_FOLDED_EFFECTS_STATE_ROOT" <<'FOLDED_BOOTSTRAP'
import os
import pwd
import sys
from contextlib import ExitStack


def bootstrap_folded_directories(state_root: str, uid: int, gid: int) -> None:
    volume, state_name = os.path.split(state_root)
    if volume != "/app/effects-data" or state_name in ("", ".", ".."):
        raise ValueError("state root must name one directory under /app/effects-data")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    with ExitStack() as opened:
        def directory(parent: int | None, name: str, create: bool = False) -> int:
            if create:
                try:
                    os.mkdir(name, mode=0o755, dir_fd=parent)
                except FileExistsError:
                    pass
            fd = os.open(name, flags, dir_fd=parent)
            opened.callback(os.close, fd)
            return fd

        app = directory(None, "/app")
        effects = directory(app, "effects-data", create=True)
        state = directory(effects, state_name, create=True)
        owned = [effects, state]
        try:
            owned.append(directory(state, "delegation"))
        except FileNotFoundError:
            # Preflight creates fresh delegation state after the privilege drop;
            # it must still detect a missing preserved store behind an alias.
            pass
        for fd in owned:
            os.fchown(fd, uid, gid)


if __name__ == "__main__":
    try:
        runtime_user = pwd.getpwnam("omniinfra")
        bootstrap_folded_directories(
            sys.argv[1], runtime_user.pw_uid, runtime_user.pw_gid
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"[entrypoint] FOLDED_EFFECTS_STATE_REFUSED: {exc}", file=sys.stderr)
        sys.exit(2)
FOLDED_BOOTSTRAP
  fi
  # The runtime-owned secret store (OMN-17099): a named volume mounted under the
  # runtime user's home is root-owned when fresh, and the process below runs as
  # omniinfra, so it could neither read nor create the store until repaired.
  install -d -o omniinfra -g omniinfra /home/omniinfra/.omninode /home/omniinfra/.omninode/delegation
  chown -R omniinfra:omniinfra /home/omniinfra/.omninode
  exec gosu omniinfra "$0" "$@"
fi

# =============================================================================
# Deployment Identity Banner
# =============================================================================
# Print before any service initialization so operators can immediately verify
# which code is running via: docker logs <container> | head -15
#
# RUNTIME_SOURCE_HASH and COMPOSE_PROJECT are stamped at build time from
# --build-arg values passed by onex-runtime-deploy. They default to "unknown"
# when the image is built without those args (e.g. manual docker compose up).
#
# SOURCE_DIR is the installed package location inside the container.
echo "=== OmniNode Runtime ==="
echo "RUNTIME_SOURCE_HASH=${RUNTIME_SOURCE_HASH:-unknown}"
echo "COMPOSE_PROJECT=${COMPOSE_PROJECT:-unknown}"
echo "SOURCE_DIR=/app/src"
echo "BUILD_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "========================"

# =============================================================================
# Boot Preflight + Kernel Start -- ONE Python interpreter (OMN-17372)
# =============================================================================
# This script used to launch FOUR separate cold Python interpreters before the
# kernel ever started -- the schema-fingerprint stamp, the Bifrost render, the
# secret-resolver render, and then the kernel exec -- each paying a full cold
# `import omnibase_infra`. Measured on the onex-dev runtime container
# (2026-09-06, container start 21:34:27Z): 141.3 s + 51.9 s + 50.2 s + 48.6 s =
# 292.0 s of the 551 s that elapsed before the first subscription was even
# attempted. All four steps are now one warm process.
#
# Everything those steps did lives in
# `omnibase_infra.runtime.entrypoint_preflight`, in the same order, with the
# same env guards, the same 5-attempt/1s stamp retry, the same required
# (omnibase_infra) vs best-effort (omniintelligence) policy (OMN-13666), and
# the same exit codes. It then starts the CMD: the packaged `onex-runtime` console script is
# called IN-PROCESS so the kernel inherits the warm import; any other CMD is
# `os.execvp`-ed, exactly as `exec "$@"` did.
#
# `exec` here keeps the Python process as the direct child of tini, so signals
# still reach the kernel without a shell in between.

exec python -m omnibase_infra.runtime.entrypoint_preflight "$@"
