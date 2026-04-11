# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
#
# Tests for lint_topic_names.py (OMN-3188).
#
# TDD: tests written first, linter implemented second.
# Convention: onex.{kind}.{producer}.{event-slug}.v{n}

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.validation.lint_topic_names import (
    _KNOWN_PRODUCERS,
    LintResult,
    lint_topic,
    scan_contracts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_contract(
    tmp_path: Path, topics: list[str], filename: str = "contract.yaml"
) -> Path:
    """Write a minimal contract.yaml with given topics in published_events."""
    contract: dict[str, object] = {
        "name": "test-node",
        "published_events": [{"topic": t} for t in topics],
    }
    path = tmp_path / filename
    path.write_text(yaml.dump(contract), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# lint_topic — single topic validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_valid_topic_passes() -> None:
    """A well-formed topic string returns an empty violations list."""
    result = lint_topic("onex.evt.platform.validation-run-completed.v1")
    assert result.violations == []
    assert result.is_valid is True


@pytest.mark.unit
def test_invalid_kind_caught() -> None:
    """A topic with an invalid kind segment returns a violation."""
    result = lint_topic("onex.badkind.platform.foo.v1")
    assert result.is_valid is False
    assert len(result.violations) >= 1
    assert any("kind" in v.lower() for v in result.violations)


@pytest.mark.unit
def test_missing_version_caught() -> None:
    """A topic missing the version suffix returns a violation."""
    result = lint_topic("onex.evt.platform.foo")
    assert result.is_valid is False
    assert len(result.violations) >= 1


# ---------------------------------------------------------------------------
# scan_contracts — directory scanning
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scan_valid_contracts_returns_no_violations(tmp_path: Path) -> None:
    """Scanning a contracts dir with only valid topics returns no violations."""
    contracts_dir = tmp_path / "nodes"
    contracts_dir.mkdir()
    node_dir = contracts_dir / "my-node"
    node_dir.mkdir()
    _write_contract(
        node_dir, ["onex.evt.platform.intent-classified.v1"], "contract.yaml"
    )

    violations = scan_contracts(contracts_dir)
    assert violations == []


@pytest.mark.unit
def test_scan_invalid_contracts_returns_violations(tmp_path: Path) -> None:
    """Scanning a contracts dir with an invalid topic returns at least one violation."""
    contracts_dir = tmp_path / "nodes"
    contracts_dir.mkdir()
    node_dir = contracts_dir / "bad-node"
    node_dir.mkdir()
    _write_contract(node_dir, ["onex.badkind.platform.foo.v1"], "contract.yaml")

    violations = scan_contracts(contracts_dir)
    assert len(violations) >= 1
    assert any("badkind" in v.lower() or "kind" in v.lower() for v in violations)


@pytest.mark.unit
def test_scan_missing_version_returns_violations(tmp_path: Path) -> None:
    """Scanning a contract with a topic missing the version suffix returns violations."""
    contracts_dir = tmp_path / "nodes"
    contracts_dir.mkdir()
    node_dir = contracts_dir / "bad-node"
    node_dir.mkdir()
    _write_contract(node_dir, ["onex.evt.platform.foo"], "contract.yaml")

    violations = scan_contracts(contracts_dir)
    assert len(violations) >= 1


@pytest.mark.unit
def test_scan_empty_dir_returns_no_violations(tmp_path: Path) -> None:
    """Scanning a directory with no contract.yaml files returns no violations."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    violations = scan_contracts(empty_dir)
    assert violations == []


@pytest.mark.unit
def test_scan_wrong_prefix_caught(tmp_path: Path) -> None:
    """A topic not starting with 'onex.' is caught as a violation."""
    contracts_dir = tmp_path / "nodes"
    contracts_dir.mkdir()
    node_dir = contracts_dir / "bad-node"
    node_dir.mkdir()
    _write_contract(node_dir, ["custom.evt.platform.foo.v1"], "contract.yaml")

    violations = scan_contracts(contracts_dir)
    assert len(violations) >= 1


# ---------------------------------------------------------------------------
# Producer allowlist (OMN-8507)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unknown_producer_rejected() -> None:
    """Linter must reject topic strings with producer not in the known-repos allowlist."""
    result = lint_topic("onex.evt.review-bot.foo.v1")
    assert not result.is_valid
    assert any("unknown producer" in v for v in result.violations)
    assert any("review-bot" in v for v in result.violations)


@pytest.mark.unit
def test_producer_with_underscore_rejected() -> None:
    """Producer with underscore fails both producer-pattern and allowlist checks."""
    result = lint_topic("onex.evt.review_bot.foo.v1")
    assert not result.is_valid


@pytest.mark.unit
def test_known_producer_accepted() -> None:
    """Linter must accept topic strings with producer in the known-repos allowlist."""
    result = lint_topic("onex.evt.omnimarket.review-bot-foo.v1")
    assert result.is_valid, f"Expected valid but got violations: {result.violations}"


@pytest.mark.unit
def test_all_known_producers_accepted() -> None:
    """All entries in _KNOWN_PRODUCERS must produce valid 5-segment topics."""
    for producer in _KNOWN_PRODUCERS:
        result = lint_topic(f"onex.evt.{producer}.test-event.v1")
        assert result.is_valid, (
            f"Producer {producer!r} unexpectedly rejected: {result.violations}"
        )


@pytest.mark.unit
def test_known_producers_allowlist_complete() -> None:
    """_KNOWN_PRODUCERS contains all expected canonical producer segments."""
    expected = {
        "omnimarket",
        "omnibase-infra",
        "omniclaude",
        "omniintelligence",
        "omnimemory",
        "omninode",
        "omnibase-compat",
        "github",
        "platform",
    }
    assert expected.issubset(_KNOWN_PRODUCERS), (
        f"Missing producers: {expected - _KNOWN_PRODUCERS}"
    )


# ---------------------------------------------------------------------------
# LintResult model
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lint_result_is_valid_false_when_violations() -> None:
    """LintResult.is_valid is False when violations list is non-empty."""
    result = LintResult(
        topic="onex.badkind.platform.foo.v1", violations=["invalid kind"]
    )
    assert result.is_valid is False


@pytest.mark.unit
def test_lint_result_is_valid_true_when_no_violations() -> None:
    """LintResult.is_valid is True when violations list is empty."""
    result = LintResult(topic="onex.evt.platform.foo.v1", violations=[])
    assert result.is_valid is True
