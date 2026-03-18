# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for create_kafka_topics.py multi-package discovery [OMN-5371]."""

from __future__ import annotations

import importlib.metadata as ilm
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Load the script as a module
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "create_kafka_topics",
        _SCRIPTS_DIR / "create_kafka_topics.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["create_kafka_topics"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_script()
_discover_all_packages = _mod._discover_all_packages


# ---------------------------------------------------------------------------
# Tests for _discover_all_packages
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_discover_skips_unimportable_entry_point(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Entry point that raises on load() is skipped with a warning in lenient mode."""
    bad_ep = MagicMock()
    bad_ep.name = "bad_pkg"
    bad_ep.load.side_effect = ImportError("missing module")
    monkeypatch.setattr(ilm, "entry_points", lambda *, group: [bad_ep])

    result = _discover_all_packages(lenient=True)
    assert result == []
    assert "WARNING" in capsys.readouterr().err


@pytest.mark.unit
def test_discover_skips_nonexistent_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Entry point whose __path__ does not exist is skipped with a warning."""
    ep = MagicMock()
    ep.name = "missing_path_pkg"
    pkg = MagicMock()
    pkg.__path__ = [str(tmp_path / "does_not_exist")]
    ep.load.return_value = pkg
    monkeypatch.setattr(ilm, "entry_points", lambda *, group: [ep])

    result = _discover_all_packages(lenient=True)
    assert result == []
    assert "WARNING" in capsys.readouterr().err


@pytest.mark.unit
def test_discover_returns_valid_packages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Two valid entry points both appear in the result."""
    pkg_a_dir = tmp_path / "pkg_a_nodes"
    pkg_a_dir.mkdir()
    pkg_b_dir = tmp_path / "pkg_b_nodes"
    pkg_b_dir.mkdir()

    ep_a = MagicMock()
    ep_a.name = "alpha"
    pkg_a = MagicMock()
    pkg_a.__path__ = [str(pkg_a_dir)]
    ep_a.load.return_value = pkg_a

    ep_b = MagicMock()
    ep_b.name = "beta"
    pkg_b = MagicMock()
    pkg_b.__path__ = [str(pkg_b_dir)]
    ep_b.load.return_value = pkg_b

    monkeypatch.setattr(ilm, "entry_points", lambda *, group: [ep_b, ep_a])

    result = _discover_all_packages(lenient=True)
    assert len(result) == 2
    names = [name for name, _ in result]
    assert names == ["alpha", "beta"]  # sorted by name
    assert result[0][1] == pkg_a_dir
    assert result[1][1] == pkg_b_dir


@pytest.mark.unit
def test_discover_filters_by_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """--packages filter restricts which entry points are returned."""
    pkg_dir = tmp_path / "nodes"
    pkg_dir.mkdir()

    ep_a = MagicMock()
    ep_a.name = "alpha"
    pkg_a = MagicMock()
    pkg_a.__path__ = [str(pkg_dir)]
    ep_a.load.return_value = pkg_a

    ep_b = MagicMock()
    ep_b.name = "beta"
    # Should not even be loaded
    ep_b.load.side_effect = AssertionError("should not be called")

    monkeypatch.setattr(ilm, "entry_points", lambda *, group: [ep_a, ep_b])

    result = _discover_all_packages(filter_names=["alpha"], lenient=True)
    assert len(result) == 1
    assert result[0][0] == "alpha"


@pytest.mark.unit
def test_discover_exits_on_error_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In strict mode (lenient=False), unloadable entry points cause sys.exit."""
    bad_ep = MagicMock()
    bad_ep.name = "broken_pkg"
    bad_ep.load.side_effect = ImportError("missing")
    monkeypatch.setattr(ilm, "entry_points", lambda *, group: [bad_ep])

    with pytest.raises(SystemExit) as exc_info:
        _discover_all_packages(lenient=False)
    assert exc_info.value.code == 1
