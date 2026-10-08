# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
#
# Tests for lint_topic_names.py (OMN-3188).
#
# TDD: tests written first, linter implemented second.
# Convention: onex.{kind}.{producer}.{event-slug}.v{n}

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.validation.lint_topic_names import (
    _KNOWN_PRODUCERS,
    LintResult,
    lint_topic,
    scan_contracts,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "declaration",
    [
        "AIOKafkaConsumer(*TOPICS)",
        "KafkaTransport(config=config, topics=TOPICS)",
        "consumer.subscribe(topics=TOPICS)",
    ],
)
def test_mixed_consumer_namespace_fails_naming_both(
    tmp_path: Path,
    declaration: str,
) -> None:
    """P3's falsifier exercises the same CLI used by the blocking gate."""
    bare = "onex.evt.platform.node-registration.v1"
    prefixed = f"tenant-a.{bare}"
    source = tmp_path / "consumer.py"
    source.write_text(
        f"TOPICS = ({prefixed!r}, {bare!r})\n{declaration}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validation/lint_topic_names.py",
            "--scan-python",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "mixed consumer topic namespace" in result.stderr
    assert prefixed in result.stderr
    assert bare in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    [
        "AIOKafkaConsumer('onex.evt.platform.node-registration.v1', 'onex.cmd.platform.request-introspection.v1')",
        "AIOKafkaConsumer('tenant-a.onex.evt.platform.node-registration.v1', 'tenant-a.onex.cmd.platform.request-introspection.v1')",
        "KafkaTransport(config=config, topics=())",
        "AIOKafkaConsumer('tenant-a.onex.evt.platform.node-registration.v1')\nAIOKafkaConsumer('onex.evt.platform.node-registration.v1')",
        "PUBLISH_TOPICS = ('tenant-a.onex.evt.platform.node-registration.v1', 'onex.evt.platform.node-registration.v1')",
    ],
)
def test_consistent_or_separate_consumer_sets_pass(tmp_path: Path, source: str) -> None:
    from scripts.validation.lint_topic_names import scan_python

    fixture = tmp_path / "consumer.py"
    fixture.write_text(source, encoding="utf-8")
    assert scan_python(fixture) == []


@pytest.mark.unit
def test_consumer_namespace_resolves_local_constants_and_aliases(
    tmp_path: Path,
) -> None:
    from scripts.validation.lint_topic_names import scan_python

    fixture = tmp_path / "consumer.py"
    fixture.write_text(
        "from aiokafka import AIOKafkaConsumer as Consumer\n"
        "BARE = 'onex.evt.platform.node-registration.v1'\n"
        "PREFIXED = 'tenant-a.' + BARE\n"
        "def make_consumer():\n"
        "    topics: tuple[str, ...] = (PREFIXED, BARE)\n"
        "    return Consumer(*topics)\n",
        encoding="utf-8",
    )
    violations = scan_python(fixture)
    assert any("mixed consumer topic namespace" in v for v in violations)


@pytest.mark.unit
def test_contract_mixed_consumer_namespace_names_both(tmp_path: Path) -> None:
    bare = "onex.evt.platform.node-registration.v1"
    prefixed = f"tenant-a.{bare}"
    fixture = tmp_path / "contract.yaml"
    fixture.write_text(
        yaml.safe_dump({"event_bus": {"subscribe_topics": [prefixed, bare]}})
    )
    violations = scan_contracts(tmp_path)
    assert any(
        "mixed consumer topic namespace" in v and prefixed in v and bare in v
        for v in violations
    )


@pytest.mark.unit
def test_namespace_refusal_cannot_be_suppressed_by_naming_baseline(
    tmp_path: Path,
) -> None:
    bare = "onex.evt.platform.node-registration.v1"
    prefixed = f"tenant-a.{bare}"
    fixture = tmp_path / "consumer.py"
    fixture.write_text(f"AIOKafkaConsumer({prefixed!r}, {bare!r})\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(f"{bare}\n")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validation/lint_topic_names.py",
            "--scan-python",
            str(fixture),
            "--baseline",
            str(baseline),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "mixed consumer topic namespace" in result.stderr


@pytest.mark.unit
def test_missing_validation_response_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.validation.lint_topic_names import EventBusInmemory, scan_python

    async def drop_request(
        self: EventBusInmemory,
        envelope: object,
        topic: str,
        *,
        key: bytes | None = None,
    ) -> None:
        return None

    fixture = tmp_path / "consumer.py"
    fixture.write_text("AIOKafkaConsumer('onex.evt.platform.node-registration.v1')\n")
    monkeypatch.setattr(EventBusInmemory, "publish_envelope", drop_request)
    with pytest.raises(ValueError, match="Consumer validation returned no result"):
        scan_python(fixture)


@pytest.mark.unit
def test_unparseable_consumer_source_fails_closed(tmp_path: Path) -> None:
    from scripts.validation.lint_topic_names import scan_python

    fixture = tmp_path / "consumer.py"
    fixture.write_text("AIOKafkaConsumer(\n")
    assert any(
        "could not parse consumer declarations" in v for v in scan_python(fixture)
    )


@pytest.mark.unit
def test_namespace_check_is_in_blocking_ci_and_precommit() -> None:
    from scripts.ci.ci_summary_gate import STRICT_GATE_JOBS

    root = Path(__file__).resolve().parents[4]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["topic-naming-lint"]
    assert job["name"] in STRICT_GATE_JOBS
    assert not job.get("continue-on-error", False)
    runs = [step.get("run", "") for step in job["steps"]]
    assert any(
        "lint_topic_names.py --scan-python src/omnibase_infra" in run for run in runs
    )
    assert any("test_lint_topic_names.py" in run for run in runs)
    config = yaml.safe_load((root / ".pre-commit-config.yaml").read_text())
    hook = next(
        hook
        for repo in config["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "topic-naming-lint"
    )
    assert hook["entry"] == "bash scripts/validation/run_topic_lint.sh"
    assert (
        "--scan-python" in (root / "scripts/validation/run_topic_lint.sh").read_text()
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
def test_valid_projection_snapshot_topic_passes() -> None:
    """Projection snapshot topics may use a dotted snapshot name."""
    result = lint_topic("onex.snapshot.projection.cost.token_usage.v1")
    assert result.violations == []
    assert result.is_valid is True


@pytest.mark.unit
def test_scan_contracts_accepts_projection_snapshot_publish_topic(
    tmp_path: Path,
) -> None:
    """Contract scanning accepts projection snapshot publish topics."""
    contracts_dir = tmp_path / "nodes"
    contracts_dir.mkdir()
    node_dir = contracts_dir / "snapshot-node"
    node_dir.mkdir()
    contract = {
        "name": "snapshot-node",
        "event_bus": {
            "publish_topics": ["onex.snapshot.projection.cost.by_repo.v1"],
        },
    }
    (node_dir / "contract.yaml").write_text(yaml.dump(contract), encoding="utf-8")

    violations = scan_contracts(contracts_dir)
    assert violations == []


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
def test_ui_cross_renderer_producer_accepted() -> None:
    """OMN-13131: 'ui' is a canonical cross-renderer producer domain.

    The renderer effect nodes (ui.effect.*) thin-publish capability declarations
    onto onex.cmd.ui.renderer-capability-declared.v1; the producer 'ui' must be
    accepted (not rejected as 'unknown producer').
    """
    result = lint_topic("onex.cmd.ui.renderer-capability-declared.v1")
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
        "ui",
    }
    assert expected.issubset(_KNOWN_PRODUCERS), (
        f"Missing producers: {expected - _KNOWN_PRODUCERS}"
    )


@pytest.mark.unit
def test_the_deploy_agents_three_live_topics_are_accepted() -> None:
    """OMN-18816: the .201 deploy agent's producer segment is real, not a typo.

    The agent is a standalone uv sub-project at ``scripts/deploy-agent/`` with no node
    and no ``contract.yaml``, so "deploy" is not a repo name and cannot become one.
    All three strings below are live production topics declared as constants in
    ``deploy_agent.events``; the third is the one this ticket gave a reader.

    RED before the allowlist entry: ``rebuild-rejected`` was refused as
    ``unknown producer 'deploy'``, which is what blocked the consumer's contract from
    declaring the subscription that provisions the topic. The first two were reachable
    only through omnimarket's suppression baseline, whose own header forbids new
    entries -- so without this the third had no non-suppressing path at all.
    """
    for topic in (
        "onex.cmd.deploy.rebuild-requested.v1",
        "onex.evt.deploy.rebuild-completed.v1",
        "onex.evt.deploy.rebuild-rejected.v1",
    ):
        result = lint_topic(topic)
        assert result.is_valid, (
            f"{topic} rejected by the naming lint: {result.violations}. This topic is "
            "published by the deploy agent and cannot be renamed from the consumer side"
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
