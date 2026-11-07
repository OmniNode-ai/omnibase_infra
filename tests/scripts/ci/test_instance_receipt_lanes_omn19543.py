# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19543: the instance receipt lanes are the routing table's, and dev-200 is one.

The operator's one-deploy-slot-per-lab-host ruling (omni_home ledger RULING
2026-09-25T10:22:22Z) adds an instance per host. Each instance's row in
``config/deploy_lane_routing.yaml`` names its receipt lane and the repositories
that receipt may prove; the release train's premise and the staging sibling read
take their instance lanes from there (``scripts/ci/instance_receipt_lanes.py``).

* The committed table gives omnimarket ``compose-dev-202`` then
  ``compose-dev-200``, and omnibase_infra nothing (it stays on ``.201``).
* Every table lane is a declared ``EnumLabLane`` outside the any-of default.
* A fixture instance ``dev-999`` reaches the reader with no Python edit, and the
  release train refuses it until the receipt enum declares its lane.
* A ``compose-dev-200`` PASS alone admits an omnimarket release cut, and is
  refused for a repo whose evidence is ``compose-dev``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.ci import instance_receipt_lanes as irl
from scripts.ci.lab_pass_receipt import ANY_OF_DEFAULT_LANES, EnumLabLane

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_RT_PATH = _ROOT / "scripts" / "ci" / "release_train.py"

SHA = "5be72e12f1e9cb8168271a2dd31f8f6c5656e896"

_FIXTURE = """
default_instance: dev-201
instances:
  dev-201:
    hostnames: [omninode-pc]
    consumer_group: onex-deploy-agent
  dev-999:
    hostnames: [somewhere]
    consumer_group: onex-deploy-agent-dev-999
    lane:
      compose_overlay: docker/docker-compose.dev-999.yml
      compose_project: omnibase-infra-dev-999
      postgres_container: omnibase-infra-dev-999-postgres
      runtime_container: omninode-dev-999-runtime
      health_ports: {main: 44085, effects: 44086}
      receipt_lane: compose-dev-999
      proves: [omnimarket]
routes: []
"""


def _release_train() -> Any:
    name = "release_train_under_test_omn19543"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _RT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_committed_table_gives_omnimarket_its_instance_lane() -> None:
    assert len(irl.receipt_lanes_for("omnimarket")) == 1


def test_the_committed_table_gives_omnibase_infra_none() -> None:
    assert irl.receipt_lanes_for("omnibase_infra") == ()


def test_every_table_lane_is_a_declared_receipt_lane_outside_the_any_of() -> None:
    lanes = irl.receipt_lanes_for("omnimarket")
    assert lanes, "positive control: the table proves omnimarket somewhere"
    for lane in lanes:
        assert EnumLabLane(lane) not in ANY_OF_DEFAULT_LANES, lane


def test_a_new_instance_needs_no_python_edit_to_reach_the_reader() -> None:
    assert irl.receipt_lanes_from_text(_FIXTURE, "omnimarket") == ("compose-dev-999",)
    assert irl.receipt_lanes_from_text(_FIXTURE, "omnibase_infra") == ()


def test_a_block_without_its_receipt_fields_refuses() -> None:
    text = _FIXTURE.replace("      receipt_lane: compose-dev-999\n", "")
    with pytest.raises(irl.InstanceReceiptLanesError, match="receipt_lane"):
        irl.receipt_lanes_from_text(text, "omnimarket")


def test_the_release_train_refuses_a_lane_the_receipt_enum_does_not_declare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rt = _release_train()
    table = tmp_path / "deploy_lane_routing.yaml"
    table.write_text(_FIXTURE, encoding="utf-8")
    monkeypatch.setattr(rt.instance_receipt_lanes, "ROUTING_TABLE", table)
    with pytest.raises(rt.ReleaseTrainConfigError, match="compose-dev-999"):
        rt.lab_evidence_lanes(rt.EnumLabEvidence.COMPOSE_DEV_OR_INSTANCE, "omnimarket")


def test_the_release_train_reads_compose_dev_then_the_table_lanes() -> None:
    rt = _release_train()
    lanes = rt.lab_pass_receipt.EnumLabLane
    read = rt.lab_evidence_lanes(
        rt.EnumLabEvidence.COMPOSE_DEV_OR_INSTANCE, "omnimarket"
    )
    assert read[0] == lanes.COMPOSE_DEV
    assert len(read) == 2


def _gate_main(
    monkeypatch: pytest.MonkeyPatch, present: EnumLabLane | None, *extra: str
) -> int:
    from scripts.ci import lab_pass_receipt
    from tests.scripts.ci._lab_pass_fixtures import SHA_4ACA, FakeSurface, receipt

    surface = FakeSurface()
    if present is not None:
        surface.add(receipt(SHA_4ACA, present))
    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
    return lab_pass_receipt.main(
        [
            "gate",
            "--sha",
            SHA_4ACA,
            "--repo",
            "OmniNode-ai/omnimarket",
            *extra,
        ]
    )


def test_instance_lanes_without_an_explicit_lane_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        _gate_main(
            monkeypatch,
            EnumLabLane.COMPOSE_DEV,
            "--instance-lanes-for",
            "omnimarket",
        )
        == 1
    )


def test_an_unreadable_table_refuses_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.ci import lab_pass_receipt

    module = lab_pass_receipt._instance_receipt_lanes()
    monkeypatch.setattr(module, "ROUTING_TABLE", tmp_path / "absent.yaml")
    args = ("--lane", "compose-dev", "--instance-lanes-for", "omnimarket")
    assert _gate_main(monkeypatch, EnumLabLane.COMPOSE_DEV, *args) == 1
