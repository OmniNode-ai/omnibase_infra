# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""omnibase_compat is retired; no live use may remain in the repo.

The graduated symbols live in omnibase_core and omnibase_spi. This guard
scans the files that previously referenced the retired package and asserts
that no live (non-comment) token survives.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Repo-relative paths that previously referenced omnibase_compat.
_SCANNED_PATHS: list[str] = [
    "docker/Dockerfile.runtime",
    "scripts/deploy-agent/deploy_agent/executor.py",
    "src/omnibase_infra/handlers/handler_runtime_deploy.sh",
    "scripts/deploy-gateway.sh",
    ".github/workflows/docker-build.yml",
    "config/lab_proof_profiles.yaml",
    "src/omnibase_infra/nodes/node_lab_proof_plan_compute/contract.yaml",
]

# Tokens that must not appear in non-comment lines of the scanned files.
_RETIRED_TOKENS: tuple[str, ...] = (
    "omnibase_compat",
    "omnibase-compat",
    "OMNIBASE_COMPAT",
)


def _strip_whole_line_comments(text: str) -> str:
    """Drop lines whose first non-space character is '#' (whole-line comments)."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


@pytest.mark.unit
@pytest.mark.parametrize("repo_rel_path", _SCANNED_PATHS)
def test_no_omnibase_compat_live_use(repo_rel_path: str) -> None:
    path = _REPO_ROOT / repo_rel_path
    assert path.is_file(), f"missing {path}"
    text = path.read_text(encoding="utf-8")
    stripped = _strip_whole_line_comments(text)
    for token in _RETIRED_TOKENS:
        assert token not in stripped, (
            f"{repo_rel_path}: retired token {token!r} found in a non-comment line"
        )


@pytest.mark.unit
def test_strip_function_detects_token_in_code() -> None:
    """Positive control: a token in a non-comment line is detected."""
    sample = "x = 1\nARG OMNIBASE_COMPAT_REF=v1\n"
    stripped = _strip_whole_line_comments(sample)
    assert "OMNIBASE_COMPAT" in stripped


@pytest.mark.unit
def test_strip_function_ignores_comment_only_lines() -> None:
    """A token that appears only in a whole-line comment is not detected."""
    sample = "# omnibase_compat is retired\nx = 1\n"
    stripped = _strip_whole_line_comments(sample)
    for token in _RETIRED_TOKENS:
        assert token not in stripped
