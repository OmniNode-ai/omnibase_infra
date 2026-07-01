# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed proof for the topic-literal gate on topic_constants.py (OMN-13199 / A5).

OMN-13199 (phase A5) completed the un-allowlisting of
``src/omnibase_infra/event_bus/topic_constants.py`` from the local topic-literal
gate (``scripts/validation/check_topic_literals.py``). The prerequisite work
(OMN-13195/A4 narrowing the exemption, OMN-13202 deleting the last ``TOPIC_*``
literals and migrating the enum codegen to read ``runtime/topics.yaml``) left the
file holding only DLQ builder logic.

These tests lock that state in permanently:

1. ``topic_constants.py`` is NOT whole-file exempt — its basename is absent from
   ``_EXCLUDED_FILENAMES``, so the gate actually scans it.
2. The real ``topic_constants.py`` is literal-free (the gate passes on it today).
3. The gate FAILS CLOSED — reintroducing a bare topic literal under that basename
   makes ``main()`` return exit code 1, so a future regression is blocked at CI /
   pre-commit time rather than silently allowed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validation.check_topic_literals import (
    _EXCLUDED_FILENAMES,
    collect_files,
    find_topic_literals,
    main,
)

_TOPIC_CONSTANTS_BASENAME = "topic_constants.py"
# Repo root resolved relative to this test file: tests/unit/scripts/validation/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
_REAL_TOPIC_CONSTANTS = (
    _REPO_ROOT / "src" / "omnibase_infra" / "event_bus" / _TOPIC_CONSTANTS_BASENAME
)


@pytest.mark.unit
def test_topic_constants_not_whole_file_exempt() -> None:
    """topic_constants.py must not be in the gate's whole-file exemption set.

    Locks the OMN-13199/A5 outcome: the file is fully un-allowlisted, so any
    reintroduced literal is scanned (not silently skipped).
    """
    assert _TOPIC_CONSTANTS_BASENAME not in _EXCLUDED_FILENAMES


@pytest.mark.unit
def test_topic_constants_is_scanned_by_the_gate() -> None:
    """The real topic_constants.py is included in the gate's scan set."""
    if not _REAL_TOPIC_CONSTANTS.exists():
        pytest.skip(f"{_REAL_TOPIC_CONSTANTS} not found")
    src_root = _REPO_ROOT / "src"
    scanned = collect_files(src_root)
    assert _REAL_TOPIC_CONSTANTS in scanned


@pytest.mark.unit
def test_real_topic_constants_is_literal_free() -> None:
    """The real topic_constants.py holds no raw topic literals (DLQ builders only)."""
    if not _REAL_TOPIC_CONSTANTS.exists():
        pytest.skip(f"{_REAL_TOPIC_CONSTANTS} not found")
    hits = find_topic_literals(_REAL_TOPIC_CONSTANTS)
    assert hits == [], f"Unexpected raw topic literals in topic_constants.py: {hits}"


@pytest.mark.unit
def test_gate_fails_closed_on_reintroduced_literal(tmp_path: Path) -> None:
    """A reintroduced bare topic literal under topic_constants.py fails the gate.

    Copies the real file into a temporary ``src/`` tree, appends a module-level
    topic literal assignment, and asserts ``main()`` returns 1 (fail-closed).
    """
    if not _REAL_TOPIC_CONSTANTS.exists():
        pytest.skip(f"{_REAL_TOPIC_CONSTANTS} not found")

    src_root = tmp_path / "src" / "omnibase_infra" / "event_bus"
    src_root.mkdir(parents=True)
    target = src_root / _TOPIC_CONSTANTS_BASENAME
    body = _REAL_TOPIC_CONSTANTS.read_text(encoding="utf-8")
    body += (
        '\n_OMN13199_FAIL_CLOSED_PROOF = "onex.evt.omnimarket.fail-closed-proof.v1"\n'
    )
    target.write_text(body, encoding="utf-8")

    empty_baseline = tmp_path / "baseline.txt"
    empty_baseline.write_text("# no suppressions\n", encoding="utf-8")

    exit_code = main(src_dir=tmp_path / "src", baseline_path=empty_baseline)
    assert exit_code == 1, (
        "gate must FAIL when a literal is reintroduced in topic_constants.py"
    )


@pytest.mark.unit
def test_gate_passes_on_clean_topic_constants_copy(tmp_path: Path) -> None:
    """The unmodified topic_constants.py passes the gate in an isolated tree."""
    if not _REAL_TOPIC_CONSTANTS.exists():
        pytest.skip(f"{_REAL_TOPIC_CONSTANTS} not found")

    src_root = tmp_path / "src" / "omnibase_infra" / "event_bus"
    src_root.mkdir(parents=True)
    target = src_root / _TOPIC_CONSTANTS_BASENAME
    target.write_text(
        _REAL_TOPIC_CONSTANTS.read_text(encoding="utf-8"), encoding="utf-8"
    )

    empty_baseline = tmp_path / "baseline.txt"
    empty_baseline.write_text("# no suppressions\n", encoding="utf-8")

    exit_code = main(src_dir=tmp_path / "src", baseline_path=empty_baseline)
    assert exit_code == 0, (
        "gate must PASS on the clean, literal-free topic_constants.py"
    )
