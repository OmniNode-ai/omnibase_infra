# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20074: the contract graph's corpus no longer includes onex_change_control.

``contract-topic-graph`` resolves its checkout packages from a directory the
workflow populates. The workflow stopped checking the change-control repository
out, so the graph must neither look for it nor fail closed because it is absent.
These tests run the real resolver and the real graph builder over a checkout
root that holds only the packages the workflow still checks out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.validators import contract_topic_graph as graph

pytestmark = pytest.mark.integration


def _checkout_root(tmp_path: Path, packages: tuple[str, ...]) -> Path:
    for package in packages:
        (tmp_path / package / "src" / package).mkdir(parents=True)
    return tmp_path


def test_resolver_resolves_every_remaining_checkout_package_and_no_change_control(
    tmp_path: Path,
) -> None:
    root = _checkout_root(tmp_path, graph.CHECKOUT_PACKAGES)
    roots = graph.discover_contract_roots(root)
    assert set(graph.CHECKOUT_PACKAGES) <= set(roots)
    assert "onex_change_control" not in roots


def test_resolver_ignores_a_change_control_checkout_if_one_is_present(
    tmp_path: Path,
) -> None:
    root = _checkout_root(tmp_path, (*graph.CHECKOUT_PACKAGES, "onex_change_control"))
    assert "onex_change_control" not in graph.discover_contract_roots(root)


def test_graph_fails_closed_naming_only_the_packages_still_expected(
    tmp_path: Path,
) -> None:
    """Positive control: a missing checkout still refuses, and never names change control."""
    with pytest.raises(RuntimeError) as excinfo:
        graph.build_graph(checkout_root=tmp_path)
    message = str(excinfo.value)
    assert "omnimarket" in message
    assert "onex_change_control" not in message


def test_frozen_environment_resolves_the_entire_supported_graph_corpus(
    tmp_path: Path,
) -> None:
    """The retired compatibility package cannot be a required graph input."""
    roots = graph.discover_contract_roots(
        _checkout_root(tmp_path, graph.CHECKOUT_PACKAGES)
    )
    assert set(graph.GRAPH_PACKAGES) <= set(roots)
    assert "omnibase_compat" not in graph.GRAPH_PACKAGES
    graph.build_graph(roots=roots)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["node_contract_registry_reducer", "node_context_audit_dlq_effect"]
)
async def test_no_routing_contract_is_not_an_auto_wiring_drop(name: str) -> None:
    """Replay both recorded false findings through the production wiring adapter."""
    from unittest.mock import MagicMock

    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
    from omnibase_infra.runtime.auto_wiring.discovery import (
        discover_contracts_from_paths,
    )
    from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
    from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
    from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
    from omnibase_infra.validators.models import ModelTopicGraph

    path = (
        Path(__file__).resolve().parents[3]
        / "src/omnibase_infra/nodes"
        / name
        / "contract.yaml"
    )
    discovered = discover_contracts_from_paths([path])
    bus = MagicMock(spec=EventBusInmemory)
    report = await wire_from_manifest(
        ModelAutoWiringManifest(contracts=tuple(discovered.contracts)),
        MessageDispatchEngine(),
        event_bus=bus,
        environment="local",
    )
    assert len(report.results) == 1
    assert report.results[0].reason == "No handler_routing declared in contract"
    assert not report.results[0].topics_subscribed
    bus.subscribe.assert_not_called()

    node = graph.parse_contract(path, "omnibase_infra")
    assert node is not None
    assert node.subscribe_topics, "Keep the declared consumption in the graph"
    assert node.has_dispatch_wiring is None
    findings = graph.find_defects(
        ModelTopicGraph(
            nodes=(node,),
            consumers=dict.fromkeys(node.subscribe_topics, (name,)),
        )
    )
    assert not [f for f in findings if f.defect == "DECLARED_BUT_UNWIRED"]
    # Absence of auto-routing does not excuse a missing producer.
    assert {f.topic for f in findings if f.defect == "ORPHANED_CONSUMER"} == set(
        node.subscribe_topics
    )


@pytest.mark.parametrize("routing", [{}, {"handlers": []}, {"handlers": [{}]}])
def test_declared_invalid_routing_remains_a_wiring_failure(
    tmp_path: Path, routing: dict[str, object]
) -> None:
    import yaml

    from omnibase_infra.validators.models import ModelTopicGraph

    source = (
        Path(__file__).resolve().parents[3]
        / "src/omnibase_infra/nodes/node_context_audit_dlq_effect/contract.yaml"
    )
    declared_topics = yaml.safe_load(source.read_text())["event_bus"][
        "subscribe_topics"
    ]
    path = tmp_path / "nodes/node_broken/contract.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "name": "node_broken",
                "event_bus": {"subscribe_topics": declared_topics},
                "handler_routing": routing,
            }
        )
    )
    node = graph.parse_contract(path, "omnibase_infra")
    assert node is not None
    findings = graph.find_defects(ModelTopicGraph(nodes=(node,)))
    assert any(f.defect == "DECLARED_BUT_UNWIRED" for f in findings)


def test_graph_is_unconditional_and_registered_in_the_required_summary() -> None:
    import yaml

    from scripts.ci.ci_summary_gate import STRICT_GATE_JOBS

    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load(
        (root / ".github/workflows/contract-topic-graph.yml").read_text()
    )
    assert set(workflow.get("on", workflow.get(True))) == {"workflow_call"}
    caller = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())["jobs"][
        "contract-topic-graph"
    ]
    assert caller["uses"] == "./.github/workflows/contract-topic-graph.yml"
    assert "if" not in caller
    assert "needs" not in caller
    assert "contract-topic-graph / contract-topic-graph" in STRICT_GATE_JOBS
    inner = workflow["jobs"]["contract-topic-graph"]
    assert "if" not in inner
    assert "needs" not in inner
    assert "continue-on-error" not in inner


@pytest.mark.parametrize("validator_exit", [0, 23])
def test_graph_workflow_propagates_the_validator_exit(
    tmp_path: Path, validator_exit: int
) -> None:
    import os
    import subprocess

    import yaml

    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load(
        (root / ".github/workflows/contract-topic-graph.yml").read_text()
    )
    steps = workflow["jobs"]["contract-topic-graph"]["steps"]
    command = next(
        step["run"]
        for step in steps
        if "validators.contract_topic_graph" in step.get("run", "")
    )
    assert "--scope omnibase_infra" in command
    assert "--touched" not in command
    # Replay the checked-in Actions shell with a refusing/successful validator.
    uv = tmp_path / "uv"
    uv.write_text(f"#!/bin/sh\nexit {validator_exit}\n")
    uv.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-e",
            "-o",
            "pipefail",
            "-c",
            command.replace("${{ github.workspace }}", str(tmp_path)),
        ],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == validator_exit, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("absent", 2),
        ("in_progress", 2),
        ("failure", 1),
        ("cancelled", 1),
        ("skipped", 1),
        ("success", 0),
    ],
)
def test_summary_waits_for_and_requires_graph_success(
    state: str, expected: int
) -> None:
    from scripts.ci.ci_summary_gate import (
        SKIPPABLE_GATE_JOBS,
        STRICT_GATE_JOBS,
        evaluate,
    )

    name = "contract-topic-graph / contract-topic-graph"
    jobs = [
        {"name": gate, "status": "completed", "conclusion": "success", "run_attempt": 1}
        for gate in (*STRICT_GATE_JOBS, *SKIPPABLE_GATE_JOBS)
        if gate != name
    ]
    if state != "absent":
        jobs.append(
            {
                "name": name,
                "status": "in_progress" if state == "in_progress" else "completed",
                "conclusion": None if state == "in_progress" else state,
                "run_attempt": 1,
            }
        )
    code, report = evaluate(jobs)
    assert code == expected, report
