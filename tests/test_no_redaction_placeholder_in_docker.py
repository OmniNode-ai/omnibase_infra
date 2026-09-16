# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18478: no redaction placeholder, and no literal password, under ``docker/``.

The PostToolUse secret redactor (``omniclaude`` ``plugins/onex/hooks/lib/secret_redactor.py``,
wired with ``matcher: "Bash"``) rewrites Bash tool OUTPUT, substituting its
replacement token for anything shaped like a credential. It does not touch
Write/Edit content. So the token reaches a tracked file by exactly one route: an
author read text through a Bash preview and used the redacted result as the
source for an edit.

That is how ``docker/catalog/services/tenant-projection-writer.yaml`` came to bind
the token as its postgres password in ``b2b9ca87a`` -- the commit that created the
file, so there is no earlier revision holding a correct value.

Two lanes then misdiagnosed it, both by reading rather than classifying:

* A sweep grepped each file and read the OUTPUT, which the same redactor had
  rewritten -- so 10 files looked corrupted where 1 was.
* A second lane compared password segments by length and hash, found nine files
  sharing one 20-byte value, and called it a committed literal. The value is the
  string ``${POSTGRES_PASSWORD}``, which is 20 bytes.

Hence :func:`classify_password_segment`, which the scan below is built on: a DSN
password is an EXPANSION, the PLACEHOLDER, or an opaque LITERAL, decided from its
structure. A hash tells you two values differ; it does not say what either is.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = REPO_ROOT / "docker"

# Built from parts, never spelled: a literal here would be rewritten in any Bash
# preview of this file, so a reader greps the test and sees the token in place of
# the code that defines it.
REDACTION_PLACEHOLDER = "*" * 3 + "REDACTED" + "*" * 3

# A file whose purpose is to exercise the redactor may hold the token. None exists
# under docker/ today; the carve-out is declared so that adding one is a deliberate
# edit to this list rather than a silent pass.
REDACTOR_FIXTURE_FILES: frozenset[str] = frozenset()

_TEXT_SUFFIXES = frozenset(
    {".yaml", ".yml", ".env", ".sh", ".py", ".conf", ".json", ".toml", ".md"}
)

# The password segment may contain spaces: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD
# required} is a legal value. An earlier revision of this pattern excluded
# whitespace, so it stopped matching the moment the fix landed and the scan
# below passed by matching nothing at all. Stop at the @, not at a space.
_DSN = re.compile(r"postgresql://([^:@\s]+):([^@\n]+)@")
_EXPANSION = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*(?:(?::?[-?+])[^}]*)?\}$")


def classify_password_segment(segment: str) -> str:
    """Return ``EXPANSION``, ``PLACEHOLDER`` or ``LITERAL`` for a DSN password."""
    if REDACTION_PLACEHOLDER in segment:
        return "PLACEHOLDER"
    if _EXPANSION.match(segment):
        return "EXPANSION"
    return "LITERAL"


def _text_files(root: Path) -> list[Path]:
    return [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and (p.suffix in _TEXT_SUFFIXES or p.suffix == "")
    ]


def find_placeholder_hits(
    root: Path, *, allowlist: frozenset[str] = frozenset()
) -> list[str]:
    """Return ``path:line`` for every occurrence of the redaction token under ``root``."""
    hits: list[str] = []
    for path in _text_files(root):
        rel = path.relative_to(root).as_posix()
        if rel in allowlist:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if REDACTION_PLACEHOLDER in line:
                hits.append(f"{rel}:{lineno}")
    return hits


def scan_dsn_passwords(root: Path) -> tuple[list[str], int]:
    """Return non-expansion findings and the total number of DSNs examined.

    The count is returned so a caller can refuse a vacuous pass. A scan that
    matches nothing reports zero findings and reads exactly like a clean tree.
    """
    findings: list[str] = []
    examined = 0
    for path in _text_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for match in _DSN.finditer(line):
                examined += 1
                verdict = classify_password_segment(match.group(2))
                if verdict != "EXPANSION":
                    findings.append(f"{rel}:{lineno} {verdict}")
    return findings, examined


def find_non_expansion_dsn_passwords(root: Path) -> list[str]:
    """Return ``path:line verdict`` for every DSN password that is not an expansion."""
    return scan_dsn_passwords(root)[0]


@pytest.mark.unit
def test_no_redaction_placeholder_anywhere_under_docker() -> None:
    """AC2. A redaction token under ``docker/`` is redacted tool output, not a value."""
    hits = find_placeholder_hits(DOCKER_DIR, allowlist=REDACTOR_FIXTURE_FILES)
    assert hits == [], (
        "The secret redactor's replacement token is bound as a value under docker/: "
        f"{hits}. This is Bash tool output that was written into a tracked file. "
        "Recover the real value, or bind it as a ${VAR:?...} expansion."
    )


@pytest.mark.unit
def test_positive_control_planted_placeholder_is_found(tmp_path: Path) -> None:
    """The scan is proven to detect what it exists to detect.

    Without this, a scan that silently stopped matching would report zero hits and
    read exactly like a clean tree.
    """
    planted = tmp_path / "catalog" / "services" / "planted.yaml"
    planted.parent.mkdir(parents=True)
    planted.write_text(
        f"hardcoded_env:\n  DB_URL: 'postgresql://postgres:{REDACTION_PLACEHOLDER}@h:5432/d'\n",
        encoding="utf-8",
    )
    assert find_placeholder_hits(tmp_path) == ["catalog/services/planted.yaml:2"]
    assert find_non_expansion_dsn_passwords(tmp_path) == [
        "catalog/services/planted.yaml:2 PLACEHOLDER"
    ]


@pytest.mark.unit
def test_catalog_service_dsn_passwords_are_all_expansions() -> None:
    """AC1. Every catalog DSN resolves its password from the environment."""
    findings, examined = scan_dsn_passwords(DOCKER_DIR / "catalog" / "services")
    assert examined >= 20, (
        f"only {examined} DSN bindings matched under docker/catalog/services -- the "
        "scan is not reading the catalog, so a zero-findings result proves nothing"
    )
    assert findings == [], (
        "A catalog manifest spells a DSN password instead of expanding one: "
        f"{findings}. Use ${{POSTGRES_PASSWORD:?POSTGRES_PASSWORD required}}, the "
        "shape docker-compose.infra.yml:143 uses."
    )


@pytest.mark.unit
def test_classify_password_segment_distinguishes_the_three_cases() -> None:
    """The 20-byte expansion that was misread as a password is classified correctly."""
    assert classify_password_segment("${POSTGRES_PASSWORD}") == "EXPANSION"
    assert classify_password_segment("${POSTGRES_PASSWORD:?required}") == "EXPANSION"
    assert (
        classify_password_segment("${POSTGRES_PASSWORD:-test-password}") == "EXPANSION"
    )
    assert classify_password_segment(REDACTION_PLACEHOLDER) == "PLACEHOLDER"
    assert classify_password_segment("test-password") == "LITERAL"
