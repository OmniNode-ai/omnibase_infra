# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21485: the C15, C16 and C28 proofs grade the stability lane, not dev.

On 2026-10-27 the dev lane went unhealthy overnight (broker memory), and
chain-canary.yml (C15), chain-canary-c16-receipt-identity.yml (C16) and
chain-canary-c28-consumer-flow.yml (C28) all went red with no defect in what
they measure, because all three graded the dev lane. The operator ruling of
that day: the proofs connect to the stability lane, the lane the team works
in, not to dev.

These tests parse the three workflows and the contracts they run, and fail on
anything that would point a proof back at the dev lane:

* the lane list each proof walks is the stability overlay
  (``vars.LAB_PROOF_LANES_JSON``), never the lab overlay the dev lane is
  declared in (``vars.LAB_LANES_JSON``), and the resolver only accepts an
  entry that declares itself a stability lane (``lane_role=stability``);
* the broker each proof reads is the one the resolved lane declares
  (``bus_lane``), never a literal ``dev`` lane of the CI bus lane table, and
  C28 reads the containers of the resolved lane's own compose project;
* no workflow value and no endpoint block of a contract names a dev-lane
  port, compose project or container, or the dev lane by name.

Lab host names and addresses never appear in this repository at all: the
public-repo hygiene gate refuses them, so the lanes' addresses live only in
the overlay variable and this file does not spell them either.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_lane import (
    BASE_COMPOSE_PROJECT,
    ModelConsumerFlowLane,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
PROOF_WORKFLOWS = (
    "chain-canary.yml",
    "chain-canary-c16-receipt-identity.yml",
    "chain-canary-c28-consumer-flow.yml",
)
#: The proofs that read the CI bus lane table; C28 reaches its broker through
#: the resolved lane's own broker container.
BUS_TABLE_WORKFLOWS = PROOF_WORKFLOWS[:2]
BOARD_PROBE_CONTRACT = (
    ROOT / "src/omnibase_infra/nodes/node_board_probe_effect/contract.yaml"
)
CHAIN_CANARY_CONTRACT = (
    ROOT / "src/omnibase_infra/nodes/node_chain_canary_effect/contract.yaml"
)

STABILITY_OVERLAY = "${{ vars.LAB_PROOF_LANES_JSON }}"
STABILITY_MATCH = "lane_role=stability"

#: A dev-lane port, compose project or container, as the dev lane declares
#: them (docker/docker-compose.infra.yml, the base the dev lane runs as-is),
#: the dev lane by name, or the overlay the dev lane is declared in.
DEV_LANE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "a dev-lane port",
        re.compile(r":(?:8085|8086|8090|8098|3002|19092|5436)\b"),
    ),
    (
        "the dev compose project",
        re.compile(rf"compose_project\s*[=:]\s*['\"]?{BASE_COMPOSE_PROJECT}(?![-\w])"),
    ),
    (
        "a dev-lane container",
        re.compile(
            r"\b(?:omninode-runtime(?:-effects)?|omnimarket-projection-api|"
            r"omnibase-infra-(?:redpanda|postgres))(?![-\w])"
        ),
    ),
    ("the dev lane by name", re.compile(r"\bdev[- ]lane\b", re.IGNORECASE)),
    ("the dev lane overlay", re.compile(r"\bvars\.LAB_LANES_JSON\b")),
)


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _strings(node: Any, where: str = "") -> Iterator[tuple[str, str]]:
    """Every string value (and mapping key) under ``node``, with its path.

    YAML comments are dropped by the parser, so history told in a comment
    is not a target; every value a run can act on is.
    """
    if isinstance(node, str):
        yield where, node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield f"{where}.{key}", str(key)
            yield from _strings(value, f"{where}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _strings(value, f"{where}[{index}]")


def _dev_lane_hits(node: Any) -> list[str]:
    return [
        f"{where}: {label} in {text[:120]!r}"
        for where, text in _strings(node)
        for label, pattern in DEV_LANE_PATTERNS
        if pattern.search(text)
    ]


def _resolver_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("uses") == "./.github/actions/resolve-lab-lane"
    ]


@pytest.mark.parametrize("name", PROOF_WORKFLOWS)
def test_proofs_target_stability_lane_overlay(name: str) -> None:
    workflow = _load(WORKFLOWS / name)
    steps = _resolver_steps(workflow)
    assert steps, f"{name}: no lane resolver step"
    for step in steps:
        given = step["with"]
        assert given["lanes-json"] == STABILITY_OVERLAY, (
            f"{name}: walks {given['lanes-json']}, not the stability overlay"
        )
        assert STABILITY_MATCH in given.get("match", "").split(), (
            f"{name}: a lane that is not a stability lane is eligible"
        )
        if name in BUS_TABLE_WORKFLOWS:
            assert "bus_lane" in given["require"].split(), (
                f"{name}: the resolved lane need not declare its broker"
            )


@pytest.mark.parametrize("name", BUS_TABLE_WORKFLOWS)
def test_proofs_target_stability_lane_broker(name: str) -> None:
    workflow = _load(WORKFLOWS / name)
    values = "\n".join(value for _where, value in _strings(workflow))
    # The CI bus lane a proof reads is the resolved lane's own declaration.
    assert "LANE_BUS_LANE" in values, f"{name}: the broker is not the lane's own"
    for where, value in _strings(workflow):
        assert not re.search(r"load_lane_\w+\([^)]*['\"]dev['\"]", value, re.S), (
            f"{name}{where}: reads the dev lane of the bus lane table"
        )
        assert not re.search(r"\|\|\s*'dev'", value), (
            f"{name}{where}: falls back to the dev bus lane"
        )
    dispatch = (workflow.get(True) or workflow.get("on") or {}).get(
        "workflow_dispatch"
    ) or {}
    for input_name, spec in (dispatch.get("inputs") or {}).items():
        assert str(spec.get("default", "")) != "dev", (
            f"{name}: dispatch input {input_name} defaults to the dev lane"
        )


@pytest.mark.parametrize("name", PROOF_WORKFLOWS)
def test_proofs_target_stability_lane_names_no_dev_lane(name: str) -> None:
    hits = _dev_lane_hits(_load(WORKFLOWS / name))
    assert not hits, f"{name} names the dev lane:\n" + "\n".join(hits)


def test_proofs_target_stability_lane_contracts() -> None:
    board = _load(BOARD_PROBE_CONTRACT)
    hits = _dev_lane_hits({"consumer_flow_endpoints": board["consumer_flow_endpoints"]})
    canary_hits = [
        hit
        for hit in _dev_lane_hits(_load(CHAIN_CANARY_CONTRACT))
        # Prose that recounts the dev-lane history is not a target; only a
        # port, compose project or container would be.
        if "the dev lane by name" not in hit
    ]
    assert not hits + canary_hits, "a proof contract names the dev lane:\n" + (
        "\n".join(hits + canary_hits)
    )


def test_proofs_target_stability_lane_c28_reads_the_resolved_lane() -> None:
    """C28 requires the lane's compose project and hands it to the node."""
    workflow = _load(WORKFLOWS / "chain-canary-c28-consumer-flow.yml")
    (step,) = _resolver_steps(workflow)
    assert "compose_project" in step["with"]["require"].split()
    prepare = next(
        s
        for s in workflow["jobs"]["c28-consumer-flow"]["steps"]
        if s.get("name") == "Prepare the consumer-flow node request"
    )
    assert prepare["env"]["LANE_COMPOSE_PROJECT"] == (
        "${{ fromJSON(needs.resolve-lane.outputs.lane_json).compose_project }}"
    )
    assert "compose_project: $compose_project" in prepare["run"]


def _declared_lane_containers() -> Iterator[tuple[str, set[str]]]:
    """Every compose lane file's project and container names, read from disk.

    Read as text: the overlays use compose's own tags (``!override``), which a
    plain YAML loader refuses.
    """
    for path in sorted((ROOT / "docker").glob("docker-compose.*.yml")):
        text = path.read_text(encoding="utf-8")
        project = re.search(r"^name:\s*['\"]?([\w-]+)", text, re.MULTILINE)
        if project is None or not project.group(1).startswith(
            f"{BASE_COMPOSE_PROJECT}-"
        ):
            continue
        names = set(
            re.findall(r"^\s+container_name:\s*['\"]?([\w-]+)", text, re.MULTILINE)
        )
        yield project.group(1), names


def test_proofs_target_stability_lane_c28_containers_follow_the_compose_project() -> (
    None
):
    """The derived names are the ones each lane's compose file declares."""
    checked = 0
    for project, names in _declared_lane_containers():
        lane = ModelConsumerFlowLane.from_compose_project(project)
        if not {lane.broker_container, *lane.runtime_containers} <= names:
            continue  # a partial overlay (a bus-only or helper file), not a lane
        wanted = {
            *lane.runtime_containers,
            lane.projection_api_container,
            lane.broker_container,
            lane.postgres_container,
        }
        assert wanted <= names, f"{project}: {sorted(wanted - names)} undeclared"
        checked += 1
    assert checked >= 3, "no lane compose file was checked"


def test_proofs_target_stability_lane_c28_refuses_a_foreign_project() -> None:
    with pytest.raises(ValueError, match="is not"):
        ModelConsumerFlowLane.from_compose_project("some-other-project")


@pytest.mark.parametrize("env", ["stability", "local"])
def test_proofs_target_stability_lane_c28_kinds_follow_the_lane_environment(
    env: str,
) -> None:
    """A consumer group is ``{env}.{service}.{node}...``; C28 reads the lane's own."""
    from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_constants import (
        KINDS,
    )
    from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_consumer_flow import (
        kind_rows,
    )
    from scripts.ci import c28_consumer_flow_probe as script

    for _kind, prefix, topic, _counter in KINDS:
        row = {"consumer_group": f"{env}.{prefix}1.0.0", "topic": topic}
        assert kind_rows([row], prefix, topic) == [row]
        assert script.kind_rows([row], prefix, topic) == [row]
        other = {"consumer_group": f"{env}.unrelated.consume.1.0.0", "topic": topic}
        assert kind_rows([other], prefix, topic) == []
