# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/check_dockerfile_pins.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_dockerfile_pins.py"


def _load_module() -> Any:
    mod_name = "check_dockerfile_pins"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_mod = _load_module()


@pytest.mark.unit
def test_parse_dockerfile_accepts_uv_with_retry_multiline_install(
    tmp_path: Path,
) -> None:
    dockerfile = tmp_path / "Dockerfile.runtime"
    dockerfile.write_text(
        """
RUN --mount=type=cache,target=/root/.cache/uv \\
    uv-with-retry pip install --no-deps \\
    "omninode-claude>=0.9.0,<2.0.0" \\
    "omninode-memory>=0.9.0,<2.0.0"
""",
        encoding="utf-8",
    )

    entries = _mod._parse_dockerfile(dockerfile)

    assert {(entry.package, entry.specifier, entry.no_deps) for entry in entries} == {
        ("omninode-claude", ">=0.9.0,<2.0.0", True),
        ("omninode-memory", ">=0.9.0,<2.0.0", True),
    }


@pytest.mark.unit
def test_parse_dockerfile_preserves_uv_pip_install_support(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile.runtime"
    dockerfile.write_text(
        'RUN uv pip install --no-deps "torch>=2.6.0,<3.0.0"\n',
        encoding="utf-8",
    )

    entries = _mod._parse_dockerfile(dockerfile)

    assert [(entry.package, entry.specifier, entry.no_deps) for entry in entries] == [
        ("torch", ">=2.6.0,<3.0.0", True),
    ]
