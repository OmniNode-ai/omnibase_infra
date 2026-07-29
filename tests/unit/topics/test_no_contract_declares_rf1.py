# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Ratchet: no contract or topic manifest may declare replication_factor 1.

OMN-15395 removed the module-level RF1 default, but eleven contracts in this
repo *declared* ``topic_config.replication_factor: 1`` outright — the provisioner
would have honoured them and minted RF1 topics on MSK even with the default
gone. Those declarations were removed; this test is the mechanism that keeps
them gone, because a rule nobody can execute is not enforcement.

Removing the declaration (rather than raising it to 2) is deliberate: a
single-broker self-hosted Redpanda cannot create an RF2 topic, so a hard RF2
declaration would break local and CI provisioning. Undeclared means "the
environment policy decides" — RF1 on self-hosted, refused on managed staging
until the owning contract states what durability it needs.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra"


def _walk_topic_configs(node: Any, source: Path) -> Iterator[tuple[Path, Any, Any]]:
    """Yield ``(source, topic, topic_config)`` for every declared topic config."""
    if isinstance(node, dict):
        topic_config = node.get("topic_config")
        if isinstance(topic_config, dict):
            yield source, node.get("topic"), topic_config
        for value in node.values():
            yield from _walk_topic_configs(value, source)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_topic_configs(item, source)


def _yaml_sources() -> list[Path]:
    sources = sorted(SRC_ROOT.glob("nodes/*/contract.yaml"))
    sources += sorted(SRC_ROOT.rglob("topics.yaml"))
    return sources


def test_no_contract_declares_replication_factor_one() -> None:
    """A contract-declared RF1 is an MSK durability defect — keep it at zero."""
    offenders: list[str] = []
    for source in _yaml_sources():
        try:
            document = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:  # pragma: no cover - malformed YAML is a
            pytest.fail(f"{source} is not parseable YAML: {exc}")
        for path, topic, topic_config in _walk_topic_configs(document, source):
            if topic_config.get("replication_factor") == 1:
                offenders.append(f"{path.relative_to(SRC_ROOT.parent.parent)}: {topic}")

    assert not offenders, (
        "Contracts declaring replication_factor: 1 will mint RF1 topics on the "
        "managed cluster, which is unrecoverable data loss on a single broker "
        "failure (AWS_KAFKA_HIGH_RISK_CONFIG_RF_EQUALS_ONE). Remove the "
        "declaration to let the environment policy decide, or declare >= 2 if "
        "the topic only ever lives on a multi-broker cluster. Offenders: "
        f"{offenders}"
    )


def test_contract_sources_were_actually_scanned() -> None:
    """Guard the guard: an empty scan would make the ratchet vacuously green."""
    sources = _yaml_sources()
    assert len(sources) > 50, f"expected the full contract tree, scanned {len(sources)}"
    with_configs = [
        topic
        for source in sources
        for _, topic, _ in _walk_topic_configs(
            yaml.safe_load(source.read_text(encoding="utf-8")), source
        )
    ]
    assert with_configs, "no topic_config blocks found — the walker is broken"
