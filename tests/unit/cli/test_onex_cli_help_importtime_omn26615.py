# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""AC3 evidence: a profile of `onex --help` names the slow imports gone (OMN-26615)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = REPO_ROOT / "docs" / "OMN-26615-onex-help-importtime.md"
SLOW_PREFIXES = ("omnibase_infra", "omnimarket", "redis", "fastapi", "aiohttp")


def test_ac3_artifact_exists() -> None:
    assert ARTIFACT.is_file(), f"missing profile artifact: {ARTIFACT}"


def test_ac3_artifact_names_before_and_after() -> None:
    text = ARTIFACT.read_text()
    assert "Median wall time before:" in text
    assert "Median wall time after:" in text
    assert "Slowest imports before" in text
    assert "Slowest imports after" in text
    assert text.count("import time:") >= 2


def test_ac3_slow_imports_gone_from_onex_help() -> None:
    text = ARTIFACT.read_text()
    after_section = text.split("## Slowest imports after", 1)[-1].split("\n## ", 1)[0]
    after_modules = set()
    for line in after_section.splitlines():
        match = re.fullmatch(r"import time:\s*\d+\s*\|\s*\d+\s*\|\s*(\S+)\s*", line)
        if match is not None:
            after_modules.add(match.group(1).strip())
    offending = sorted(
        name
        for name in after_modules
        if any(name == p or name.startswith(p + ".") for p in SLOW_PREFIXES)
    )
    assert not offending, f"slow imports still present in after profile: {offending}"
