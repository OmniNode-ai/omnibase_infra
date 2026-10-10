# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Live repository references use the renamed knowledge_base slug (OMN-20837)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
_ROOT = Path(__file__).resolve().parents[3]
_OLD_SLUG = re.compile(r"OmniNode-ai/knowledge-base(?!-internal)")


def _live_surfaces() -> Iterator[tuple[Path, str]]:
    paths = [_ROOT / "README.md"]
    for directory in ("config", "scripts", "src", "docker", ".github"):
        paths.extend(sorted((_ROOT / directory).rglob("*")))
    for path in paths:
        relative = path.relative_to(_ROOT)
        if any(part in {"tests", "fixtures", "__pycache__"} for part in relative.parts):
            continue
        if path.name.startswith("test_") or path.name.endswith(".captured"):
            continue
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if "\x00" in content:
            continue
        yield relative, content


def test_live_surfaces_name_no_old_knowledge_base_slug() -> None:
    matches: list[str] = []
    for relative, content in _live_surfaces():
        for lineno, line in enumerate(content.splitlines(), start=1):
            for _ in _OLD_SLUG.finditer(line):
                matches.append(f"{relative.as_posix()}:{lineno}: {line}")
    assert not matches, "Old knowledge-base slug in live surfaces:\n" + "\n".join(
        matches
    )


@pytest.mark.parametrize(
    "relative",
    ["config/lab_proof_profiles.yaml", "config/runner_fleet.yaml"],
)
def test_live_surfaces_name_the_renamed_slug(relative: str) -> None:
    content = (_ROOT / relative).read_text(encoding="utf-8")
    assert "knowledge_base" in content
    if relative == "config/lab_proof_profiles.yaml":
        assert "OmniNode-ai/knowledge_base" in content
