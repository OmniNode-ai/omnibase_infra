# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17389 -- dispatch-failure sink subscription gate.

node_ledger_projection_compute subscribed to the quarantine sink it was being
republished into, creating a closed self-feeding loop:

  1. Runtime consumes a record from onex.dlq.omnibase-infra.quarantine.v1
  2. service_kernel refuses it -- `.dlq.` topics have no valid message category
  3. DLQQuarantineProducer republishes the failure back to the SAME topic
  4. Back to step 1, forever

126 republishes of one payload in 2h (~one every 57s). Topic HWM growing at
~300/hr. chain-canary.yml RED on every 2-hourly run; verdict=quarantined.

Fixed by OMN-18013 (Jonah, 2026-09-07): quarantine topic removed from
node_ledger_projection_compute subscribe_topics. This gate prevents recurrence.

The dispatch-failure quarantine sink is derived at test time from
build_dlq_topic -- the same factory used by DLQQuarantineProducer -- so a
renamed quarantine topic is caught without editing this file.

RED ON PARENT (eee49719c): node_ledger_projection_compute subscribed to
onex.dlq.omnibase-infra.quarantine.v1. PASSES on OMN-18013 and every commit
since that leaves the subscription removed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.event_bus.topic_constants import build_dlq_topic

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "omnibase_infra"


def _dispatch_failure_sinks() -> frozenset[str]:
    """Return dispatch-failure sink topics that contracts must not consume."""
    return frozenset({build_dlq_topic("quarantine")})


def _scan_subscribers(
    paths: list[Path],
    sink_topics: frozenset[str],
) -> list[tuple[str, Path, str]]:
    """Return (contract_name, path, matched_topic) for every path that subscribes to a sink topic."""
    violators: list[tuple[str, Path, str]] = []
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        name: str = data.get("name", path.parent.name)
        subscribe_topics: list[str] = (data.get("event_bus") or {}).get(
            "subscribe_topics"
        ) or []
        for topic in subscribe_topics:
            if topic in sink_topics:
                violators.append((name, path, topic))
    return violators


def _repo_contract_paths() -> list[Path]:
    return sorted(
        p
        for p in SRC.rglob("contract.yaml")
        if ".venv" not in p.parts and "site-packages" not in p.parts
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _format_violators(violators: list[tuple[str, Path, str]]) -> str:
    return (
        "Contracts subscribing to a dispatch-failure sink (see OMN-17389):\n"
        + "\n".join(
            f"  {name} ({_display_path(path)}): {topic}"
            for name, path, topic in violators
        )
    )


@pytest.mark.unit
def test_no_contract_subscribes_to_dispatch_failure_sink() -> None:
    """Zero contracts subscribe to any dispatch-failure sink topic.

    RED ON PARENT: node_ledger_projection_compute subscribed to the quarantine
    sink it was being republished into (OMN-17389). This gate asserts the class
    of bug is structurally impossible: no contract may subscribe to a topic in
    the authoritative dispatch-failure sink set.
    """
    violators = _scan_subscribers(_repo_contract_paths(), _dispatch_failure_sinks())
    assert violators == [], _format_violators(violators)


@pytest.mark.unit
def test_gate_fires_when_subscriber_injected(tmp_path: Path) -> None:
    """Positive control: the gate is not vacuously green.

    Writes a synthetic contract that subscribes to the quarantine sink and
    asserts the scanner returns it as a violator. The source tree is never
    mutated; all writes go to tmp_path.
    """
    quarantine_topic = next(iter(_dispatch_failure_sinks()))
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        "name: synthetic_bad_node\n"
        "event_bus:\n"
        "  subscribe_topics:\n"
        f"    - {quarantine_topic!r}\n",
        encoding="utf-8",
    )
    violators = _scan_subscribers([contract_path], _dispatch_failure_sinks())
    assert len(violators) == 1
    name, path, topic = violators[0]
    assert name == "synthetic_bad_node"
    assert path == contract_path
    assert topic == quarantine_topic


@pytest.mark.unit
def test_failure_message_lists_multiple_violators_from_tmp_paths(
    tmp_path: Path,
) -> None:
    """Positive control: failure output is robust for non-repo temp paths."""
    quarantine_topic = next(iter(_dispatch_failure_sinks()))
    other_topic = "onex.events.omnibase-infra.allowed.v1"
    first = tmp_path / "one" / "contract.yaml"
    second = tmp_path / "two" / "contract.yaml"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(
        "name: synthetic_bad_node_one\n"
        "event_bus:\n"
        "  subscribe_topics:\n"
        f"    - {quarantine_topic!r}\n"
        f"    - {other_topic!r}\n",
        encoding="utf-8",
    )
    second.write_text(
        "name: synthetic_bad_node_two\n"
        "event_bus:\n"
        "  subscribe_topics:\n"
        f"    - {quarantine_topic!r}\n",
        encoding="utf-8",
    )

    violators = _scan_subscribers([first, second], _dispatch_failure_sinks())
    message = _format_violators(violators)

    assert len(violators) == 2
    assert "synthetic_bad_node_one" in message
    assert "synthetic_bad_node_two" in message
    assert str(first) in message
    assert str(second) in message
    assert quarantine_topic in message
    assert other_topic not in message
