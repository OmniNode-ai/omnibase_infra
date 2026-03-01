#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Sync ~/.omnibase/.env non-bootstrap keys to Infisical.

This script reads ``~/.omnibase/.env``, strips out bootstrap-only keys
(those required to *start* Infisical itself), and forwards the rest to
``seed-infisical.py --execute`` so they land in Infisical.

It is designed to be safe to call frequently (e.g. from a zsh hook after
editing .env) because of its 5-guard architecture:

1. **INFISICAL_ADDR guard** — skip silently if ``INFISICAL_ADDR`` is not set
   or empty. This preserves the opt-in behaviour: local dev without Infisical
   is unaffected.
2. **Env file guard** — fail loudly (exit 1) if ``~/.omnibase/.env`` is
   missing. The sync cannot proceed without the source of truth.
3. **uv guard** — warn and skip (exit 0) if the ``uv`` binary is not on
   PATH. Avoids crashing shells that don't have uv installed.
4. **Throttle guard** — skip (exit 0) if the last successful sync was
   within :data:`THROTTLE_SECONDS` (5 minutes). Prevents hammering
   Infisical when the hook fires many times in quick succession.
5. **flock guard** — acquire a non-blocking exclusive advisory lock
   (:func:`fcntl.flock`) so that concurrent invocations (e.g. multiple
   terminal tabs all sourcing .env simultaneously) don't race each other.

Bootstrap keys excluded from sync:

* ``POSTGRES_PASSWORD``
* ``INFISICAL_ENCRYPTION_KEY``
* ``INFISICAL_AUTH_SECRET``
* ``INFISICAL_ADDR``
* ``INFISICAL_CLIENT_ID``
* ``INFISICAL_CLIENT_SECRET``
* ``INFISICAL_PROJECT_ID``

Usage (normally invoked automatically from a shell hook or post-save script)::

    uv run python scripts/sync-omnibase-env.py
    uv run python scripts/sync-omnibase-env.py --dry-run   # default: False

.. versionadded:: 0.10.2
    Created as part of OMN-3243.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sync-omnibase-env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Bootstrap keys that must NOT be synced to Infisical.
#: These are the keys required to *bootstrap* Infisical itself, so pushing
#: them into Infisical would create a circular dependency.
BOOTSTRAP_KEYS: frozenset[str] = frozenset(
    {
        "POSTGRES_PASSWORD",
        "INFISICAL_ENCRYPTION_KEY",
        "INFISICAL_AUTH_SECRET",
        "INFISICAL_ADDR",
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
        "INFISICAL_PROJECT_ID",
    }
)

#: Default path to the ``~/.omnibase/.env`` file.
DEFAULT_ENV_FILE: Path = Path.home() / ".omnibase" / ".env"

#: Throttle window in seconds (5 minutes).
THROTTLE_SECONDS: int = 300

#: Path to the last-sync timestamp file.
LAST_SYNC_FILE: Path = Path.home() / ".cache" / "omnibase" / "sync-omnibase-env.last"

#: Path to the advisory lock file.
_DEFAULT_LOCK_FILE: Path = (
    Path.home() / ".cache" / "omnibase" / "sync-omnibase-env.lock"
)


# ---------------------------------------------------------------------------
# Guard 1: INFISICAL_ADDR
# ---------------------------------------------------------------------------


def check_infisical_addr() -> bool:
    """Return True if ``INFISICAL_ADDR`` is set and non-empty.

    When False, the caller should skip the sync silently.
    """
    addr = os.environ.get("INFISICAL_ADDR", "")
    if not addr:
        logger.info("INFISICAL_ADDR is not set — skipping sync (opt-in required)")
        return False
    return True


# ---------------------------------------------------------------------------
# Guard 2: Env file
# ---------------------------------------------------------------------------


def check_env_file(env_file: Path) -> bool:
    """Assert that *env_file* exists.

    Returns:
        True if the file exists.

    Raises:
        FileNotFoundError: If the file does not exist. The message always
            includes ``.env`` so callers and tests can match on it.
    """
    if not env_file.is_file():
        raise FileNotFoundError(
            f"Required .env file not found: {env_file}. "
            "Cannot sync without the source of truth."
        )
    return True


# ---------------------------------------------------------------------------
# Guard 3: uv
# ---------------------------------------------------------------------------


def check_uv_available() -> bool:
    """Return True if the ``uv`` binary is on PATH.

    Logs a warning and returns False if uv is absent. Does not raise, so
    the caller can degrade gracefully.
    """
    if shutil.which("uv") is None:
        logger.warning(
            "uv binary not found on PATH — skipping Infisical sync. "
            "Install uv (https://docs.astral.sh/uv/) to enable syncing."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Guard 4: Throttle
# ---------------------------------------------------------------------------


def check_throttle(last_sync_file: Path) -> bool:
    """Return True if the throttle window has passed (sync should proceed).

    Returns False if the last sync was within :data:`THROTTLE_SECONDS` ago,
    meaning the caller should skip this run.

    Corrupt or missing timestamp files are treated as "no prior sync" so the
    sync is allowed to proceed.
    """
    if not last_sync_file.is_file():
        return True
    try:
        last_ts = float(last_sync_file.read_text().strip())
    except (ValueError, OSError):
        logger.warning(
            "Could not read last-sync timestamp from %s — allowing sync",
            last_sync_file,
        )
        return True
    now = time.time()
    elapsed = now - last_ts
    if elapsed < 0:
        # Clock skew: timestamp is in the future — treat as "no prior sync" to
        # avoid bypassing the throttle window with a large negative elapsed value.
        logger.warning(
            "Last-sync timestamp %.0f is in the future (now=%.0f, delta=%.0f s) "
            "— treating as no prior sync",
            last_ts,
            now,
            elapsed,
        )
        return True
    if elapsed < THROTTLE_SECONDS:
        logger.info(
            "Last sync was %.0f seconds ago (throttle window: %d s) — skipping",
            elapsed,
            THROTTLE_SECONDS,
        )
        return False
    return True


def write_last_sync_timestamp(last_sync_file: Path) -> None:
    """Write the current epoch timestamp to *last_sync_file*.

    Creates parent directories as needed.
    """
    last_sync_file.parent.mkdir(parents=True, exist_ok=True)
    last_sync_file.write_text(str(time.time()))
    logger.debug("Wrote last-sync timestamp to %s", last_sync_file)


# ---------------------------------------------------------------------------
# Guard 5: flock
# ---------------------------------------------------------------------------


def acquire_flock(lock_file: Path) -> object | None:
    """Acquire an advisory exclusive non-blocking lock on *lock_file*.

    Creates *lock_file* (and its parents) if they do not exist.

    Returns:
        An open file object whose descriptor holds the lock, or ``None`` if
        the lock could not be acquired (another process has it).
    """
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_file, "w")  # noqa: SIM115 — intentionally kept open to hold the flock
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except OSError as exc:
        fd.close()
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            # Lock is held by another process — expected during concurrent runs.
            logger.info("Another sync is already running — skipping (flock busy)")
            return None
        # Unexpected I/O error (e.g. filesystem failure) — propagate.
        raise


def release_flock(fd: object) -> None:
    """Release the advisory lock and close *fd*."""
    if fd is None:
        return
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]
        fd.close()  # type: ignore[union-attr]
    except OSError as exc:
        logger.warning("Error releasing flock: %s", exc)


# ---------------------------------------------------------------------------
# Bootstrap key filtering
# ---------------------------------------------------------------------------


def filter_bootstrap_keys(env_vars: dict[str, str]) -> dict[str, str]:
    """Return *env_vars* with all :data:`BOOTSTRAP_KEYS` removed."""
    return {k: v for k, v in env_vars.items() if k not in BOOTSTRAP_KEYS}


# ---------------------------------------------------------------------------
# Core sync logic
# ---------------------------------------------------------------------------


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse a ``.env`` file into key=value pairs.

    Delegates to the shared ``_infisical_util._parse_env_file`` helper so
    quoting / comment-stripping behaviour stays consistent across all scripts.
    """
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from _infisical_util import _parse_env_file as _util_parse

    return _util_parse(env_path)


def _run_seed_infisical(
    filtered_vars: dict[str, str],
    seed_script: Path,
    dry_run: bool,
) -> int:
    """Invoke seed-infisical.py with the filtered key=value pairs.

    Args:
        filtered_vars: The env vars to sync (bootstrap keys already removed).
        seed_script: Absolute path to seed-infisical.py.
        dry_run: When True, passes ``--dry-run`` (no writes to Infisical).

    Returns:
        The exit code from seed-infisical.py.
    """
    if not filtered_vars:
        logger.info("No keys to sync after filtering bootstrap keys")
        return 0

    cmd = [
        "uv",
        "run",
        "python",
        str(seed_script),
        "--set-values",
        "--create-missing-keys",
    ]
    if not dry_run:
        cmd.append("--execute")

    # Pass the filtered vars as KEY=VALUE environment so seed-infisical can
    # pick them up via --import-env-from-environ (if supported) or via env.
    # Simpler: write to a temp file and pass --import-env.

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".env", delete=False, prefix="sync-omnibase-env-"
    ) as tmp:
        tmp_path = Path(tmp.name)
        for key, value in filtered_vars.items():
            # Wrap value in single quotes to survive shell splitting edge cases.
            escaped = value.replace("'", "'\\''")
            tmp.write(f"{key}='{escaped}'\n")

    try:
        cmd += ["--import-env", str(tmp_path)]
        logger.info(
            "Running: %s (%d keys, dry_run=%s)",
            " ".join(cmd[:5]) + " ...",
            len(filtered_vars),
            dry_run,
        )
        result = subprocess.run(cmd, check=False, timeout=30)
        return result.returncode
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main(
    env_file: Path | None = None,
    last_sync_file: Path | None = None,
    lock_file: Path | None = None,
    dry_run: bool = False,
) -> int:
    """Execute the env→Infisical sync with all 5 guards applied.

    Args:
        env_file: Path to the source ``.env`` file. Defaults to
            :data:`DEFAULT_ENV_FILE`.
        last_sync_file: Path to the throttle timestamp file. Defaults to
            :data:`LAST_SYNC_FILE`.
        lock_file: Path to the flock file. Defaults to an internal default.
        dry_run: When True, sync is simulated but no writes happen.

    Returns:
        Exit code: 0 for success or skip, non-zero for errors.
    """
    env_file = env_file or DEFAULT_ENV_FILE
    last_sync_file = last_sync_file or LAST_SYNC_FILE
    lock_file = lock_file or _DEFAULT_LOCK_FILE

    # -------------------------------------------------------------------
    # Guard 1: INFISICAL_ADDR
    # -------------------------------------------------------------------
    if not check_infisical_addr():
        return 0

    # -------------------------------------------------------------------
    # Guard 2: Env file
    # -------------------------------------------------------------------
    try:
        check_env_file(env_file)
    except FileNotFoundError as exc:
        logger.exception("%s", exc)
        return 1

    # -------------------------------------------------------------------
    # Guard 3: uv
    # -------------------------------------------------------------------
    if not check_uv_available():
        return 0

    # -------------------------------------------------------------------
    # Guard 4: Throttle
    # -------------------------------------------------------------------
    if not check_throttle(last_sync_file):
        return 0

    # -------------------------------------------------------------------
    # Guard 5: flock (non-blocking — skip if another sync is running)
    # -------------------------------------------------------------------
    fd = acquire_flock(lock_file)
    if fd is None:
        return 0

    try:
        # ---------------------------------------------------------------
        # Parse .env and filter bootstrap keys
        # ---------------------------------------------------------------
        env_vars = _parse_env_file(env_file)
        filtered = filter_bootstrap_keys(env_vars)
        logger.info(
            "Parsed %d vars from %s, %d after filtering bootstrap keys",
            len(env_vars),
            env_file,
            len(filtered),
        )

        # ---------------------------------------------------------------
        # Invoke seed-infisical.py
        # ---------------------------------------------------------------
        seed_script = Path(__file__).resolve().parent / "seed-infisical.py"
        rc = _run_seed_infisical(filtered, seed_script, dry_run=dry_run)
        if rc == 0:
            write_last_sync_timestamp(last_sync_file)
            logger.info("Sync complete (dry_run=%s)", dry_run)
        else:
            logger.error("seed-infisical.py exited with code %d", rc)
        return rc
    finally:
        release_flock(fd)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Path to the .env file (default: ~/.omnibase/.env)",
    )
    parser.add_argument(
        "--last-sync-file",
        type=Path,
        default=LAST_SYNC_FILE,
        help="Path to the throttle timestamp file",
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=_DEFAULT_LOCK_FILE,
        help="Path to the advisory lock file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate sync without writing to Infisical",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    sys.exit(
        main(
            env_file=args.env_file,
            last_sync_file=args.last_sync_file,
            lock_file=args.lock_file,
            dry_run=args.dry_run,
        )
    )
