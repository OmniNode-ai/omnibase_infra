# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for workspace-mode sibling lock-pin resolution + preflight (OMN-12989).

Defect under test: the 2026-06-11 stability bootstrap crash was caused by a
workspace-mode rebuild that vendored omnibase_infra 0.37.0-dev (a 13-day-stale
worktree) even though omnimarket dev's uv.lock pins infra 0.38.1. The build
ignored the lock pin and shipped a downgraded sibling, which removed the
OMN-12501 Protocol-quarantine guard and turned a latent contract defect into a
fatal crash.

These tests encode the ratchet contract:
  (a) expected sibling pins are resolved from the consuming repo's uv.lock,
  (b) a fail-fast preflight aborts on version regression / SHA mismatch,
  (c) the comparison block is exported for build-provenance.json.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "runtime_build" / "resolve_workspace_pins.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "resolve_workspace_pins", str(MODULE_PATH)
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec so dataclass KW_ONLY detection can resolve the module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# Minimal uv.lock fragment mirroring omnimarket dev's pin structure (git-rev source).
_LOCK_TEXT = """
version = 1
requires-python = ">=3.12"

[[package]]
name = "omnibase-compat"
version = "0.5.1"
source = { git = "https://github.com/OmniNode-ai/omnibase_compat.git?rev=4d887307aae34d9d40d389ba91070cb411ce3df5#4d887307aae34d9d40d389ba91070cb411ce3df5" }

[[package]]
name = "omnibase-infra"
version = "0.38.1"
source = { git = "https://github.com/OmniNode-ai/omnibase_infra.git?rev=e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59#e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59" }

[[package]]
name = "onex-change-control"
version = "0.9.0"
source = { git = "https://github.com/OmniNode-ai/onex_change_control.git?rev=4877d3c223517cb0c7e1eca462ba0f4d38916314" }

[[package]]
name = "omnimarket"
version = "0.4.3"
source = { editable = "." }
"""


def test_module_exists() -> None:
    assert MODULE_PATH.exists(), f"resolve_workspace_pins.py not found at {MODULE_PATH}"


def test_parse_lock_pins_extracts_version_and_rev(tmp_path: Path) -> None:
    mod = _load_module()
    lock = tmp_path / "uv.lock"
    lock.write_text(_LOCK_TEXT, encoding="utf-8")

    pins = mod.parse_lock_pins(lock)

    assert pins["omnibase-infra"].version == "0.38.1"
    assert pins["omnibase-infra"].rev == "e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59"
    assert pins["omnibase-compat"].version == "0.5.1"
    assert pins["omnibase-compat"].rev == "4d887307aae34d9d40d389ba91070cb411ce3df5"
    assert pins["onex-change-control"].rev == "4877d3c223517cb0c7e1eca462ba0f4d38916314"


def test_parse_lock_pins_missing_lock_raises(tmp_path: Path) -> None:
    mod = _load_module()
    with pytest.raises(FileNotFoundError):
        mod.parse_lock_pins(tmp_path / "does-not-exist.lock")


def test_compare_pin_exact_match_is_ok() -> None:
    mod = _load_module()
    result = mod.compare_pin(
        package="omnibase-infra",
        expected_version="0.38.1",
        expected_rev="e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59",
        actual_version="0.38.1",
        actual_rev="e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59",
    )
    assert result.match is True
    assert result.status == "exact"


def test_compare_pin_forward_advance_is_ok() -> None:
    """Canonical clone may be AT OR AHEAD of the lock pin (dev advances) — allowed."""
    mod = _load_module()
    result = mod.compare_pin(
        package="omnibase-infra",
        expected_version="0.38.1",
        expected_rev="e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59",
        actual_version="0.38.3",
        actual_rev="20a42cbf0000000000000000000000000000aaaa",
    )
    assert result.match is True
    assert result.status == "ahead"


def test_compare_pin_regression_is_violation() -> None:
    """The exact 06-11 crash: vendored 0.37.0 against a 0.38.1 lock pin."""
    mod = _load_module()
    result = mod.compare_pin(
        package="omnibase-infra",
        expected_version="0.38.1",
        expected_rev="e2dbdc950540df8bc59ca4370b2d4a0f5b8d6c59",
        actual_version="0.37.0",
        actual_rev="2c1d672f0000000000000000000000000000bbbb",
    )
    assert result.match is False
    assert result.status == "regression"


def test_assert_pins_satisfied_raises_on_regression() -> None:
    mod = _load_module()
    comparisons = [
        mod.PinComparison(
            package="omnibase-infra",
            expected_version="0.38.1",
            expected_rev="e2dbdc95",
            actual_version="0.37.0",
            actual_rev="2c1d672f",
            match=False,
            status="regression",
        )
    ]
    with pytest.raises(mod.WorkspacePinError) as exc:
        mod.assert_pins_satisfied(comparisons)
    msg = str(exc.value)
    assert "omnibase-infra" in msg
    assert "0.38.1" in msg  # expected
    assert "0.37.0" in msg  # actual


def test_assert_pins_satisfied_passes_when_all_match() -> None:
    mod = _load_module()
    comparisons = [
        mod.PinComparison(
            package="omnibase-infra",
            expected_version="0.38.1",
            expected_rev="e2dbdc95",
            actual_version="0.38.3",
            actual_rev="20a42cbf",
            match=True,
            status="ahead",
        )
    ]
    # Must not raise.
    mod.assert_pins_satisfied(comparisons)


def test_read_repo_version_from_pyproject(tmp_path: Path) -> None:
    mod = _load_module()
    repo = tmp_path / "omnibase_infra"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "omnibase-infra"\nversion = "0.38.3"\n',
        encoding="utf-8",
    )
    assert mod.read_repo_version(repo) == "0.38.3"


def test_read_repo_version_missing_pyproject_raises(tmp_path: Path) -> None:
    mod = _load_module()
    with pytest.raises(FileNotFoundError):
        mod.read_repo_version(tmp_path / "no-repo")


def test_build_comparisons_detects_staged_regression(tmp_path: Path) -> None:
    """End-to-end: a staged sibling tree older than the lock pin is flagged."""
    mod = _load_module()

    # Consuming repo (omnimarket) with the dev lock pinning infra 0.38.1.
    omnimarket = tmp_path / "omnimarket"
    omnimarket.mkdir()
    (omnimarket / "uv.lock").write_text(_LOCK_TEXT, encoding="utf-8")
    (omnimarket / "pyproject.toml").write_text(
        '[project]\nname = "omnimarket"\nversion = "0.4.3"\n', encoding="utf-8"
    )

    # Staged infra sibling at a STALE 0.37.0 (the crash condition).
    infra = tmp_path / "omnibase_infra"
    infra.mkdir()
    (infra / "pyproject.toml").write_text(
        '[project]\nname = "omnibase-infra"\nversion = "0.37.0"\n', encoding="utf-8"
    )

    comparisons = mod.build_comparisons(
        lock_path=omnimarket / "uv.lock",
        siblings={"omnibase-infra": infra},
    )
    infra_cmp = next(c for c in comparisons if c.package == "omnibase-infra")
    assert infra_cmp.status == "regression"
    assert infra_cmp.match is False

    with pytest.raises(mod.WorkspacePinError):
        mod.assert_pins_satisfied(comparisons)
