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


# ---------------------------------------------------------------------------
# npm global install pin coverage (OMN-13248: codex + claude CLIs)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_npm_global_installs_extracts_scoped_pinned_packages(
    tmp_path: Path,
) -> None:
    dockerfile = tmp_path / "Dockerfile.runtime"
    dockerfile.write_text(
        """
RUN set -eu; \\
    npm install -g \\
        @openai/codex@0.141.0 \\
        @anthropic-ai/claude-code@2.1.181; \\
    npm cache clean --force
""",
        encoding="utf-8",
    )

    entries = _mod._parse_npm_global_installs(dockerfile)

    # The trailing `npm cache clean --force` must NOT be parsed as install targets.
    assert {(e.package, e.version) for e in entries} == {
        ("@openai/codex", "0.141.0"),
        ("@anthropic-ai/claude-code", "2.1.181"),
    }


@pytest.mark.unit
def test_parse_npm_global_installs_supports_short_i_and_long_global_flags(
    tmp_path: Path,
) -> None:
    dockerfile = tmp_path / "Dockerfile.runtime"
    dockerfile.write_text(
        "RUN npm i --global @openai/codex@0.141.0\n",
        encoding="utf-8",
    )

    entries = _mod._parse_npm_global_installs(dockerfile)

    assert [(e.package, e.version) for e in entries] == [("@openai/codex", "0.141.0")]


@pytest.mark.unit
def test_check_npm_exact_pin_accepts_exact_version() -> None:
    entry = _mod.NpmPinEntry(package="@openai/codex", version="0.141.0", line_number=1)
    assert _mod._check_npm_exact_pin(entry) is None


@pytest.mark.unit
def test_check_npm_exact_pin_rejects_unpinned_package() -> None:
    entry = _mod.NpmPinEntry(package="@openai/codex", version="", line_number=1)
    err = _mod._check_npm_exact_pin(entry)
    assert err is not None
    assert "not pinned to an exact version" in err


@pytest.mark.unit
def test_check_npm_exact_pin_rejects_dist_tag() -> None:
    entry = _mod.NpmPinEntry(
        package="@anthropic-ai/claude-code", version="latest", line_number=1
    )
    err = _mod._check_npm_exact_pin(entry)
    assert err is not None
    assert "non-exact pin" in err


@pytest.mark.unit
def test_real_dockerfile_pins_codex_and_claude_exactly() -> None:
    """The shipped Dockerfile.runtime must pin both coding-agent CLIs exactly."""
    dockerfile = _REPO_ROOT / "docker" / "Dockerfile.runtime"
    entries = _mod._parse_npm_global_installs(dockerfile)

    by_pkg = {e.package: e.version for e in entries}
    assert by_pkg.get("@openai/codex") == "0.141.0"
    assert by_pkg.get("@anthropic-ai/claude-code") == "2.1.181"
    # And every npm global install in the real Dockerfile passes the pin gate.
    for entry in entries:
        assert _mod._check_npm_exact_pin(entry) is None
