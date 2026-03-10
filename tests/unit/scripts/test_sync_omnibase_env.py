# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for sync-omnibase-env.py script (OMN-3243).

Tests cover all 5 guards:
1. INFISICAL_ADDR guard — skip sync if INFISICAL_ADDR is not set or empty
2. Env file guard — fail clearly if ~/.omnibase/.env does not exist
3. uv guard — degrade gracefully if uv is not available
4. Throttle guard — skip if last sync was within 5 minutes
5. flock guard — prevent concurrent runs

And bootstrap key exclusion.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Script has hyphenated name so cannot use normal import; add scripts dir
# to sys.path and use importlib.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))


def _import_script() -> object:
    """Import the sync-omnibase-env module via importlib."""
    import importlib.util

    script_path = _SCRIPTS_DIR / "sync-omnibase-env.py"
    spec = importlib.util.spec_from_file_location("sync_omnibase_env", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# Guard 1: INFISICAL_ADDR guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInfisicalAddrGuard:
    """INFISICAL_ADDR guard: skip sync if INFISICAL_ADDR is not set or empty."""

    def test_skips_when_infisical_addr_not_set(self, tmp_path: Path) -> None:
        """Should exit 0 (skip) when INFISICAL_ADDR is not in environment."""
        mod = _import_script()
        with patch.dict("os.environ", {}, clear=True):
            result = mod.check_infisical_addr()  # type: ignore[attr-defined]
        assert result is False, "Should return False when INFISICAL_ADDR is absent"

    def test_skips_when_infisical_addr_empty(self, tmp_path: Path) -> None:
        """Should exit 0 (skip) when INFISICAL_ADDR is set to empty string."""
        mod = _import_script()
        with patch.dict("os.environ", {"INFISICAL_ADDR": ""}, clear=False):
            result = mod.check_infisical_addr()  # type: ignore[attr-defined]
        assert result is False, "Should return False when INFISICAL_ADDR is empty"

    def test_passes_when_infisical_addr_set(self) -> None:
        """Should return True when INFISICAL_ADDR is a non-empty string."""
        mod = _import_script()
        with patch.dict(
            "os.environ", {"INFISICAL_ADDR": "http://localhost:8880"}, clear=False
        ):
            result = mod.check_infisical_addr()  # type: ignore[attr-defined]
        assert result is True, "Should return True when INFISICAL_ADDR is set"


# ---------------------------------------------------------------------------
# Guard 2: Env file guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnvFileGuard:
    """Env file guard: fail clearly if ~/.omnibase/.env does not exist."""

    def test_fails_when_env_file_missing(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError when env file does not exist."""
        mod = _import_script()
        missing = tmp_path / "nonexistent.env"
        with pytest.raises(FileNotFoundError, match=r"\.env"):
            mod.check_env_file(missing)  # type: ignore[attr-defined]

    def test_passes_when_env_file_exists(self, tmp_path: Path) -> None:
        """Should return True when env file exists."""
        mod = _import_script()
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value\n")
        result = mod.check_env_file(env_file)  # type: ignore[attr-defined]
        assert result is True, "Should return True when env file exists"


# ---------------------------------------------------------------------------
# Guard 3: uv guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUvGuard:
    """uv guard: degrade gracefully if uv is not available."""

    def test_warns_when_uv_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Should return False and warn (not crash) when uv is not found."""
        import shutil

        mod = _import_script()
        with patch.object(shutil, "which", return_value=None):
            result = mod.check_uv_available()  # type: ignore[attr-defined]
        assert result is False, "Should return False when uv is not available"

    def test_passes_when_uv_found(self) -> None:
        """Should return True when uv is on PATH."""
        import shutil

        mod = _import_script()
        with patch.object(shutil, "which", return_value="/usr/local/bin/uv"):
            result = mod.check_uv_available()  # type: ignore[attr-defined]
        assert result is True, "Should return True when uv is available"


# ---------------------------------------------------------------------------
# Guard 4: Throttle guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestThrottleGuard:
    """Throttle guard: skip if last sync was within 5 minutes."""

    def test_skips_when_within_throttle_window(self, tmp_path: Path) -> None:
        """Should return False when last sync was <5 minutes ago."""
        mod = _import_script()
        last_sync_file = tmp_path / "sync-omnibase-env.last"
        # Write a timestamp 30 seconds ago (well within 5 min window)
        last_sync_file.write_text(str(time.time() - 30))
        result = mod.check_throttle(last_sync_file)  # type: ignore[attr-defined]
        assert result is False, "Should return False (skip) when within throttle window"

    def test_passes_when_outside_throttle_window(self, tmp_path: Path) -> None:
        """Should return True when last sync was >5 minutes ago."""
        mod = _import_script()
        last_sync_file = tmp_path / "sync-omnibase-env.last"
        # Write a timestamp 10 minutes ago
        last_sync_file.write_text(str(time.time() - 600))
        result = mod.check_throttle(last_sync_file)  # type: ignore[attr-defined]
        assert result is True, "Should return True when outside throttle window"

    def test_passes_when_no_last_sync_file(self, tmp_path: Path) -> None:
        """Should return True when no last sync file exists (first run)."""
        mod = _import_script()
        last_sync_file = tmp_path / "sync-omnibase-env.last"
        result = mod.check_throttle(last_sync_file)  # type: ignore[attr-defined]
        assert result is True, "Should return True when no last sync file exists"

    def test_passes_when_file_is_corrupt(self, tmp_path: Path) -> None:
        """Should return True (allow sync) when last sync file has corrupt content."""
        mod = _import_script()
        last_sync_file = tmp_path / "sync-omnibase-env.last"
        last_sync_file.write_text("not-a-number")
        result = mod.check_throttle(last_sync_file)  # type: ignore[attr-defined]
        assert result is True, "Should return True (allow sync) when file is corrupt"

    def test_passes_when_timestamp_is_in_future(self, tmp_path: Path) -> None:
        """Should return True (allow sync) when last sync timestamp is in the future.

        Clock skew can produce a future timestamp. Treating it as "no prior sync"
        prevents the negative elapsed value from bypassing the throttle window.
        """
        mod = _import_script()
        last_sync_file = tmp_path / "sync-omnibase-env.last"
        # Write a timestamp 60 seconds in the future (simulates clock skew)
        last_sync_file.write_text(str(time.time() + 60))
        result = mod.check_throttle(last_sync_file)  # type: ignore[attr-defined]
        assert result is True, (
            "Should return True (allow sync) when timestamp is future"
        )

    def test_throttle_window_is_5_minutes(self, tmp_path: Path) -> None:
        """Should use exactly 5-minute (300s) throttle window."""
        mod = _import_script()
        assert mod.THROTTLE_SECONDS == 300  # type: ignore[attr-defined]

    def test_writes_timestamp_on_success(self, tmp_path: Path) -> None:
        """Should write current timestamp to last sync file on success."""
        mod = _import_script()
        last_sync_file = tmp_path / "sync-omnibase-env.last"
        before = time.time()
        mod.write_last_sync_timestamp(last_sync_file)  # type: ignore[attr-defined]
        after = time.time()

        assert last_sync_file.exists(), "Timestamp file should be created"
        written = float(last_sync_file.read_text().strip())
        assert before <= written <= after, "Written timestamp should be current time"


# ---------------------------------------------------------------------------
# Guard 5: flock guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFlockGuard:
    """flock guard: prevent concurrent runs."""

    def test_acquires_lock_when_available(self, tmp_path: Path) -> None:
        """Should acquire lock file successfully when not locked."""
        mod = _import_script()
        lock_file = tmp_path / "sync-omnibase-env.lock"
        fd = mod.acquire_flock(lock_file)  # type: ignore[attr-defined]
        try:
            assert fd is not None, "Should return file descriptor on lock acquisition"
        finally:
            mod.release_flock(fd)  # type: ignore[attr-defined]

    def test_lock_file_is_created(self, tmp_path: Path) -> None:
        """Should create the lock file on disk."""
        mod = _import_script()
        lock_file = tmp_path / "sync-omnibase-env.lock"
        fd = mod.acquire_flock(lock_file)  # type: ignore[attr-defined]
        try:
            assert lock_file.exists(), "Lock file should exist after acquire"
        finally:
            mod.release_flock(fd)  # type: ignore[attr-defined]

    def test_returns_none_when_already_locked(self, tmp_path: Path) -> None:
        """Should return None (non-blocking) when lock is held by another fd."""
        import fcntl

        mod = _import_script()
        lock_file = tmp_path / "sync-omnibase-env.lock"

        # Manually hold an exclusive lock on the file
        lock_file.touch()
        with open(lock_file, "w") as holder:
            try:
                fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Now try to acquire via the module — should return None
                fd = mod.acquire_flock(lock_file)  # type: ignore[attr-defined]
                assert fd is None, "Should return None when lock is already held"
            finally:
                fcntl.flock(holder.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Bootstrap key exclusion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBootstrapKeyExclusion:
    """Bootstrap keys must NOT be synced to Infisical."""

    def test_all_bootstrap_keys_excluded(self) -> None:
        """All 7 bootstrap keys must be present in BOOTSTRAP_KEYS constant."""
        mod = _import_script()
        expected = {
            "POSTGRES_PASSWORD",
            "INFISICAL_ENCRYPTION_KEY",
            "INFISICAL_AUTH_SECRET",
            "INFISICAL_ADDR",
            "INFISICAL_CLIENT_ID",
            "INFISICAL_CLIENT_SECRET",
            "INFISICAL_PROJECT_ID",
        }
        assert expected <= set(mod.BOOTSTRAP_KEYS), (  # type: ignore[attr-defined]
            f"Missing bootstrap keys: {expected - set(mod.BOOTSTRAP_KEYS)}"
        )

    def test_filter_removes_bootstrap_keys(self) -> None:
        """filter_bootstrap_keys should remove all bootstrap keys from a dict."""
        mod = _import_script()
        env_vars = {
            "POSTGRES_PASSWORD": "secret",
            "INFISICAL_CLIENT_ID": "client-id",
            "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
            "SOME_OTHER_KEY": "some-value",
        }
        filtered = mod.filter_bootstrap_keys(env_vars)  # type: ignore[attr-defined]
        assert "POSTGRES_PASSWORD" not in filtered
        assert "INFISICAL_CLIENT_ID" not in filtered
        assert "KAFKA_BOOTSTRAP_SERVERS" in filtered
        assert "SOME_OTHER_KEY" in filtered

    def test_filter_with_empty_dict(self) -> None:
        """filter_bootstrap_keys should handle empty dict gracefully."""
        mod = _import_script()
        result = mod.filter_bootstrap_keys({})  # type: ignore[attr-defined]
        assert result == {}, "Empty dict should return empty dict"

    def test_filter_with_all_bootstrap_keys(self) -> None:
        """filter_bootstrap_keys should return empty dict when all keys are bootstrap."""
        mod = _import_script()
        env_vars = {
            "POSTGRES_PASSWORD": "secret",
            "INFISICAL_ENCRYPTION_KEY": "enc-key",
            "INFISICAL_AUTH_SECRET": "auth-secret",
            "INFISICAL_ADDR": "http://localhost:8880",
            "INFISICAL_CLIENT_ID": "client-id",
            "INFISICAL_CLIENT_SECRET": "client-secret",
            "INFISICAL_PROJECT_ID": "project-id",
        }
        result = mod.filter_bootstrap_keys(env_vars)
        assert result == {}, "All bootstrap keys should be filtered out"


# ---------------------------------------------------------------------------
# Timestamp file path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLastSyncFilePath:
    """Verify the default last-sync file path constant."""

    def test_last_sync_file_path(self) -> None:
        """Default LAST_SYNC_FILE should be under ~/.cache/omnibase/."""
        mod = _import_script()
        path = mod.LAST_SYNC_FILE  # type: ignore[attr-defined]
        assert isinstance(path, Path), "LAST_SYNC_FILE should be a Path"
        assert str(path).endswith("sync-omnibase-env.last"), (
            "LAST_SYNC_FILE should end with sync-omnibase-env.last"
        )
        assert ".cache/omnibase" in str(path), (
            "LAST_SYNC_FILE should be under ~/.cache/omnibase/"
        )


# ---------------------------------------------------------------------------
# Main entry point integration test (smoke test with all guards mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMainEntryPoint:
    """Smoke tests for the main() function guard ordering."""

    def test_main_skips_when_no_infisical_addr(self, tmp_path: Path) -> None:
        """main() should exit 0 (no-op) when INFISICAL_ADDR is not set."""
        mod = _import_script()
        with patch.dict("os.environ", {}, clear=True):
            # Should not raise, should return gracefully
            result = mod.main(  # type: ignore[attr-defined]
                env_file=tmp_path / ".env",
                last_sync_file=tmp_path / "sync.last",
                lock_file=tmp_path / "sync.lock",
                dry_run=True,
            )
        assert result == 0, "main() should return 0 when INFISICAL_ADDR guard skips"

    def test_main_fails_when_env_file_missing(self, tmp_path: Path) -> None:
        """main() should return non-zero when env file is missing."""
        mod = _import_script()
        with patch.dict(
            "os.environ", {"INFISICAL_ADDR": "http://localhost:8880"}, clear=False
        ):
            result = mod.main(  # type: ignore[attr-defined]
                env_file=tmp_path / "nonexistent.env",
                last_sync_file=tmp_path / "sync.last",
                lock_file=tmp_path / "sync.lock",
                dry_run=True,
            )
        assert result != 0, "main() should return non-zero when env file is missing"

    def test_main_skips_when_within_throttle(self, tmp_path: Path) -> None:
        """main() should exit 0 when within throttle window."""
        mod = _import_script()
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value\n")
        last_sync_file = tmp_path / "sync.last"
        last_sync_file.write_text(str(time.time() - 30))

        with patch.dict(
            "os.environ", {"INFISICAL_ADDR": "http://localhost:8880"}, clear=False
        ):
            import shutil

            with patch.object(shutil, "which", return_value="/usr/local/bin/uv"):
                result = mod.main(  # type: ignore[attr-defined]
                    env_file=env_file,
                    last_sync_file=last_sync_file,
                    lock_file=tmp_path / "sync.lock",
                    dry_run=True,
                )
        assert result == 0, "main() should return 0 when within throttle window"
