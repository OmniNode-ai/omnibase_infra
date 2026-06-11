# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/runtime_build/check_sibling_lock_pins.py [OMN-12977].

Recurrence guard for the 2026-06-11 stability bootstrap crash: the workspace-mode
image build vendored omnibase_infra 0.37.0-dev (pre-OMN-12501 guard) while
omnimarket dev's uv.lock pinned omnibase-infra 0.38.1 @ e2dbdc95. The build
ignored the lock and shipped a 13-day-stale sibling.

These tests pin the lock-parsing + pin-comparison contract so the preflight can
fail-fast on any future expected-vs-actual mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "runtime_build"
    / "check_sibling_lock_pins.py"
)

sys.path.insert(0, str(SCRIPT_PATH.parent))

from check_sibling_lock_pins import (
    ModelActualPin,
    ModelLockPin,
    ModelPinComparison,
    check_pins,
    compare_pins,
    parse_lock_pins,
)

# A minimal but faithful slice of omnimarket dev's uv.lock as it stood on
# 2026-06-11: omnibase-infra/omnibase-core are git-pinned with rev; omnibase-spi
# is a registry (PyPI) pin with no git rev.
LOCK_FIXTURE = """\
[[package]]
name = "omnibase-core"
version = "0.44.0"
source = { git = "https://github.com/OmniNode-ai/omnibase_core.git?rev=c97c2c9a45c5fb0def5fb7dacfd5f01278bb9f55#c97c2c9a45c5fb0def5fb7dacfd5f01278bb9f55" }
dependencies = [
    { name = "pydantic" },
]

[[package]]
name = "omnibase-infra"
version = "0.38.1"
source = { git = "https://github.com/OmniNode-ai/omnibase_infra.git?rev=e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59#e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59" }
dependencies = [
    { name = "a2a-sdk" },
]

[[package]]
name = "omnibase-spi"
version = "0.20.6"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "omnimarket"
version = "0.4.3"
source = { editable = "." }
dependencies = [
    { name = "omnibase-infra", git = "https://github.com/OmniNode-ai/omnibase_infra.git?rev=e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59" },
    { name = "omnibase-compat", git = "https://github.com/OmniNode-ai/omnibase_compat.git?rev=4d887307aae34d9d40d389ba91070cb411ce3df5" },
]

[[package]]
name = "unrelated-package"
version = "9.9.9"
source = { registry = "https://pypi.org/simple" }
"""


@pytest.mark.unit
class TestParseLockPins:
    def test_extracts_git_pinned_infra_version_and_rev(self) -> None:
        pins = parse_lock_pins(LOCK_FIXTURE, packages=["omnibase-infra"])
        assert "omnibase-infra" in pins
        pin = pins["omnibase-infra"]
        assert pin.version == "0.38.1"
        assert pin.git_rev == "e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59"

    def test_extracts_git_pinned_core_version_and_rev(self) -> None:
        pins = parse_lock_pins(LOCK_FIXTURE, packages=["omnibase-core"])
        pin = pins["omnibase-core"]
        assert pin.version == "0.44.0"
        assert pin.git_rev == "c97c2c9a45c5fb0def5fb7dacfd5f01278bb9f55"

    def test_registry_pin_has_no_git_rev(self) -> None:
        pins = parse_lock_pins(LOCK_FIXTURE, packages=["omnibase-spi"])
        pin = pins["omnibase-spi"]
        assert pin.version == "0.20.6"
        assert pin.git_rev is None

    def test_only_requested_packages_returned(self) -> None:
        pins = parse_lock_pins(
            LOCK_FIXTURE, packages=["omnibase-infra", "omnibase-core"]
        )
        assert set(pins) == {"omnibase-infra", "omnibase-core"}

    def test_editable_package_with_git_deps_has_no_own_rev(self) -> None:
        # Regression: omnimarket is source = { editable = "." } but its
        # dependencies carry ?rev= for git deps. A block-wide rev search would
        # wrongly attribute a dependency's rev to omnimarket. Its own rev is None.
        pins = parse_lock_pins(LOCK_FIXTURE, packages=["omnimarket"])
        assert pins["omnimarket"].version == "0.4.3"
        assert pins["omnimarket"].git_rev is None

    def test_missing_package_raises(self) -> None:
        # Fail-fast: a requested foundation package absent from the lock is a
        # build-composition error, not something to silently skip.
        with pytest.raises(KeyError):
            parse_lock_pins(LOCK_FIXTURE, packages=["omnibase-infra", "does-not-exist"])


@pytest.mark.unit
class TestComparePins:
    def test_version_and_rev_match_is_ok(self) -> None:
        expected = ModelLockPin(
            package="omnibase-infra",
            version="0.38.1",
            git_rev="e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59",
        )
        actual = ModelActualPin(
            package="omnibase-infra",
            version="0.38.1",
            git_sha="e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59",
        )
        result = compare_pins(expected, actual)
        assert isinstance(result, ModelPinComparison)
        assert result.matches is True
        assert result.mismatch_reason == ""

    def test_stale_version_is_mismatch(self) -> None:
        # The exact 2026-06-11 failure: vendored 0.37.0 vs locked 0.38.1.
        expected = ModelLockPin(
            package="omnibase-infra",
            version="0.38.1",
            git_rev="e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59",
        )
        actual = ModelActualPin(
            package="omnibase-infra",
            version="0.37.0",
            git_sha="2c1d672f00000000000000000000000000000000",
        )
        result = compare_pins(expected, actual)
        assert result.matches is False
        assert "0.38.1" in result.mismatch_reason
        assert "0.37.0" in result.mismatch_reason

    def test_version_match_but_sha_drift_is_mismatch(self) -> None:
        expected = ModelLockPin(
            package="omnibase-infra",
            version="0.38.1",
            git_rev="e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59",
        )
        actual = ModelActualPin(
            package="omnibase-infra",
            version="0.38.1",
            git_sha="ffffffffffffffffffffffffffffffffffffffff",
        )
        result = compare_pins(expected, actual)
        assert result.matches is False
        assert "ffffffff" in result.mismatch_reason

    def test_registry_pin_only_checks_version(self) -> None:
        # spi is a registry pin: no git rev to compare, version equality decides.
        expected = ModelLockPin(package="omnibase-spi", version="0.20.6", git_rev=None)
        actual = ModelActualPin(
            package="omnibase-spi",
            version="0.20.6",
            git_sha="anything-here-is-ignored",
        )
        result = compare_pins(expected, actual)
        assert result.matches is True

    def test_comparison_serializes_for_provenance(self) -> None:
        expected = ModelLockPin(
            package="omnibase-infra",
            version="0.38.1",
            git_rev="e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59",
        )
        actual = ModelActualPin(
            package="omnibase-infra", version="0.37.0", git_sha="2c1d672f"
        )
        result = compare_pins(expected, actual)
        payload = json.loads(result.model_dump_json())
        assert payload["package"] == "omnibase-infra"
        assert payload["expected_version"] == "0.38.1"
        assert payload["actual_version"] == "0.37.0"
        assert payload["matches"] is False
        assert payload["drift_direction"] == "backward"


@pytest.mark.unit
class TestDriftDirection:
    def test_stale_clone_is_backward(self) -> None:
        # The exact 2026-06-11 crash: clone 0.37.0 older than locked 0.38.1.
        expected = ModelLockPin(package="omnibase-infra", version="0.38.1")
        actual = ModelActualPin(
            package="omnibase-infra", version="0.37.0", git_sha="2c1d672f"
        )
        assert compare_pins(expected, actual).drift_direction == "backward"

    def test_newer_clone_is_forward(self) -> None:
        # Routine state: clones lead the lock until the bot bumps it.
        expected = ModelLockPin(package="omnibase-infra", version="0.38.1")
        actual = ModelActualPin(
            package="omnibase-infra", version="0.38.3", git_sha="b7c93a8e"
        )
        assert compare_pins(expected, actual).drift_direction == "forward"

    def test_match_is_none(self) -> None:
        expected = ModelLockPin(package="omnibase-spi", version="0.20.6")
        actual = ModelActualPin(package="omnibase-spi", version="0.20.6", git_sha="x")
        assert compare_pins(expected, actual).drift_direction == "none"

    def test_non_numeric_version_is_unknown(self) -> None:
        expected = ModelLockPin(package="omnimarket", version="0.4.3")
        actual = ModelActualPin(package="omnimarket", version="0.4.3-dev", git_sha="x")
        assert compare_pins(expected, actual).drift_direction == "unknown"


@pytest.mark.unit
class TestCheckPins:
    def _write_lock(self, tmp_path: Path) -> Path:
        lock = tmp_path / "uv.lock"
        lock.write_text(LOCK_FIXTURE, encoding="utf-8")
        return lock

    def _make_clone(self, tmp_path: Path, name: str, version: str) -> Path:
        import subprocess

        root = tmp_path / name
        root.mkdir()
        (root / "pyproject.toml").write_text(
            f'[project]\nname = "{name}"\nversion = "{version}"\n',
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-qm",
                "init",
            ],
            check=True,
        )
        return root

    def test_missing_lock_returns_2(self, tmp_path: Path) -> None:
        rc = check_pins(
            tmp_path / "nope.lock",
            {"omnibase-infra": tmp_path},
            output_path=None,
        )
        assert rc == 2

    def test_version_match_passes(self, tmp_path: Path) -> None:
        lock = self._write_lock(tmp_path)
        # spi is the registry pin (version-only): a matching clone passes.
        clone = self._make_clone(tmp_path, "omnibase-spi", "0.20.6")
        out = tmp_path / "out.json"
        rc = check_pins(lock, {"omnibase-spi": clone}, output_path=out)
        assert rc == 0
        payload = json.loads(out.read_text())
        assert payload["drift_count"] == 0
        assert payload["allow_drift"] is False

    def test_backward_drift_aborts(self, tmp_path: Path) -> None:
        lock = self._write_lock(tmp_path)
        clone = self._make_clone(tmp_path, "omnibase-spi", "0.19.0")
        out = tmp_path / "out.json"
        rc = check_pins(lock, {"omnibase-spi": clone}, output_path=out)
        assert rc == 1
        payload = json.loads(out.read_text())
        assert payload["drift_count"] == 1
        assert payload["comparisons"][0]["drift_direction"] == "backward"

    def test_allow_drift_records_and_proceeds(self, tmp_path: Path) -> None:
        lock = self._write_lock(tmp_path)
        clone = self._make_clone(tmp_path, "omnibase-spi", "0.19.0")
        out = tmp_path / "out.json"
        rc = check_pins(
            lock, {"omnibase-spi": clone}, output_path=out, allow_drift=True
        )
        assert rc == 0
        payload = json.loads(out.read_text())
        assert payload["allow_drift"] is True
        assert payload["drift_count"] == 1
