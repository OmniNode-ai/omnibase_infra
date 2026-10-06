# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20006: a frozen dev lane names the lane that proves in its place.

omnibase_infra#4588 froze dev-201 (receipt lane ``compose-dev``) until
2026-10-19T07:00Z as the Tech Week demo lane, so it deploys no new dev head and
no ``compose-dev`` receipt can exist for one. The release train and the staging
delivery gate both read ``compose-dev`` for omnibase_infra, so both stalled.

The substitute is declared, never inferred: the freeze block of
``config/deploy_lane_routing.yaml`` names ``substitute_instance``, that instance
proves omnibase_infra and is routed omnibase_infra's dev rebuilds, so it emits
a real ``ModelLabPassReceipt`` (PASS or FAIL, never a skip) for each merged sha
under its own receipt lane (rule 24). The train reads ``compose-dev`` then the
instance lanes; the delivery gate requires the lane the table routes the repo
to. Every green assertion has a refusing sibling.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from scripts.ci import deploy_lane_verify_route as route_reader
from scripts.ci import instance_receipt_lanes as irl
from scripts.ci.lab_pass_receipt import EnumLabLane
from tests.scripts.ci._lab_pass_fixtures import SHA_4ACA, FakeSurface, receipt

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_TABLE = _ROOT / "config" / "deploy_lane_routing.yaml"
_POLICY = _ROOT / "config" / "release_train_policy.yaml"
_RT_PATH = _ROOT / "scripts" / "ci" / "release_train.py"
_WORKFLOW = _ROOT / ".github" / "workflows" / "deliver-dev-candidate-to-staging.yml"

SHA = "40f8ec085b5be72e12f1e9cb8168271a2dd31f8f"
DURING = datetime(2026, 10, 6, 8, 0, tzinfo=UTC)
AFTER = datetime(2026, 10, 19, 7, 0, 1, tzinfo=UTC)


def _table() -> dict[str, Any]:
    return yaml.safe_load(_TABLE.read_text(encoding="utf-8"))


def _text(table: dict[str, Any]) -> str:
    return yaml.safe_dump(table, sort_keys=False)


def _rt() -> Any:
    name = "release_train_under_test_omn20006"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _RT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- the committed declaration ------------------------------------------------


def test_the_freeze_names_the_substitute_instance() -> None:
    freeze = _table()["instances"]["dev-201"]["flags"]["freeze"]
    assert freeze["substitute_instance"] == "dev-202"


def test_the_substitute_proves_omnibase_infra_and_is_routed_it() -> None:
    table = _table()
    assert "omnibase_infra" in table["instances"]["dev-202"]["lane"]["proves"]
    assert route_reader.route(table, "dev", "omnibase_infra") == "dev-202"


def test_every_other_requester_keeps_its_route() -> None:
    table = _table()
    assert route_reader.route(table, "dev", "omnimarket") == "dev-202"
    assert route_reader.route(table, "dev", "omnibase_core") == "dev-201"
    assert route_reader.route(table, "dev", None) == "dev-201"


def test_the_route_to_the_substitute_exists_only_while_a_freeze_names_it() -> None:
    """Dropping the freeze must drop the route in the same change, or dev-201
    would never again deploy omnibase_infra's dev heads."""
    table = _table()
    del table["instances"]["dev-201"]["flags"]["freeze"]
    routed = [
        row
        for row in table["routes"]
        if row["requester_repository"] == "omnibase_infra"
    ]
    assert routed, "positive control: the committed table routes omnibase_infra"
    assert irl.frozen_receipt_lanes_from_text(_text(table), DURING) == {}
    with pytest.raises(irl.InstanceReceiptLanesError, match="omnibase_infra"):
        irl.check_substitute_routes_from_text(_text(table))


def test_the_committed_table_passes_its_own_consistency_check() -> None:
    irl.check_substitute_routes_from_text(_TABLE.read_text(encoding="utf-8"))


def test_a_substitute_that_does_not_prove_the_repo_refuses() -> None:
    table = _table()
    table["instances"]["dev-202"]["lane"]["proves"] = ["omnimarket"]
    with pytest.raises(irl.InstanceReceiptLanesError, match="proves"):
        irl.check_substitute_routes_from_text(_text(table))


def test_a_substitute_naming_an_undeclared_instance_refuses() -> None:
    table = _table()
    table["instances"]["dev-201"]["flags"]["freeze"]["substitute_instance"] = "dev-9"
    with pytest.raises(irl.InstanceReceiptLanesError, match="dev-9"):
        irl.check_substitute_routes_from_text(_text(table))


# --- the readers --------------------------------------------------------------


def test_the_instance_lanes_for_omnibase_infra_are_the_substitutes() -> None:
    assert irl.receipt_lanes_for("omnibase_infra") == ("compose-dev-202",)


def test_the_frozen_lane_is_named_with_its_substitute_during_the_freeze() -> None:
    frozen = irl.frozen_receipt_lanes_from_text(_TABLE.read_text("utf-8"), DURING)
    assert set(frozen) == {"compose-dev"}
    assert frozen["compose-dev"].substitute_receipt_lane == "compose-dev-202"
    assert frozen["compose-dev"].until == datetime(2026, 10, 19, 7, tzinfo=UTC)


def test_an_expired_freeze_names_nothing() -> None:
    assert irl.frozen_receipt_lanes_from_text(_TABLE.read_text("utf-8"), AFTER) == {}


def test_the_routed_receipt_lane_is_the_substitute_for_omnibase_infra() -> None:
    assert irl.routed_receipt_lane("omnibase_infra") == "compose-dev-202"


def test_the_routed_receipt_lane_of_an_unrouted_repo_is_the_default_lanes() -> None:
    assert irl.routed_receipt_lane("omnibase_core") == "compose-dev"


def test_an_unreadable_table_refuses_the_routed_lane(tmp_path: Path) -> None:
    with pytest.raises(irl.InstanceReceiptLanesError):
        irl.routed_receipt_lane("omnibase_infra", table=tmp_path / "absent.yaml")


# --- the release train --------------------------------------------------------


def _surface(rt: Any, receipts: dict[str, str]) -> tuple[Any, Any]:
    lane_enum = rt.lab_pass_receipt.EnumLabLane
    result_enum = rt.lab_pass_receipt.EnumLabPassResult
    by_name = {
        rt.lab_pass_receipt.artifact_name(lane_enum(lane), SHA): (
            index,
            lane_enum(lane),
            result_enum(result),
        )
        for index, (lane, result) in enumerate(receipts.items(), start=1)
    }

    def list_artifacts(repo: str, name: str) -> list[dict[str, Any]]:
        assert repo == "OmniNode-ai/omnibase_infra"
        if name not in by_name:
            return []
        return [{"id": by_name[name][0], "created_at": "2026-10-06T08:00:00Z"}]

    def download_receipt(repo: str, artifact_id: int) -> Any:
        for index, lane, result in by_name.values():
            if index == artifact_id:
                return SimpleNamespace(sha=SHA, lane=lane, result=result, checks=[])
        raise AssertionError(artifact_id)

    return list_artifacts, download_receipt


def _classify(receipts: dict[str, str], now: datetime = DURING) -> tuple[Any, str]:
    rt = _rt()
    policy = rt.load_policy(_POLICY)["omnibase_infra"]
    list_artifacts, download_receipt = _surface(rt, receipts)
    return rt.classify_lab_receipt(
        "omnibase_infra",
        SHA,
        list_artifacts=list_artifacts,
        download_receipt=download_receipt,
        lanes=rt.lab_evidence_lanes(policy.lab_evidence, policy.repo),
        frozen=rt.instance_receipt_lanes.frozen_receipt_lanes(now),
    )


def test_the_train_reads_compose_dev_then_the_substitute_for_omnibase_infra() -> None:
    rt = _rt()
    policy = rt.load_policy(_POLICY)["omnibase_infra"]
    assert policy.lab_evidence is rt.EnumLabEvidence.COMPOSE_DEV_OR_INSTANCE
    lanes = rt.lab_pass_receipt.EnumLabLane
    assert rt.lab_evidence_lanes(policy.lab_evidence, "omnibase_infra") == (
        lanes.COMPOSE_DEV,
        lanes.COMPOSE_DEV_202,
    )


def test_a_substitute_pass_admits_the_cut_though_the_frozen_lane_failed() -> None:
    reason, detail = _classify({"compose-dev": "FAIL", "compose-dev-202": "PASS"})
    assert reason is None
    assert "compose-dev-202 lane" in detail


def test_a_substitute_fail_refuses_as_a_fail_never_a_skip_over_it() -> None:
    reason, _detail = _classify({"compose-dev-202": "FAIL"})
    assert reason is not None
    assert reason.value != "lab_receipt_pass"


def test_no_receipt_on_either_lane_refuses_and_names_the_freeze() -> None:
    rt = _rt()
    reason, detail = _classify({})
    assert reason is rt.EnumTrainReason.LAB_RECEIPT_ABSENT
    assert "frozen until 2026-10-19T07:00Z" in detail
    assert "compose-dev-202" in detail


def test_after_the_freeze_the_refusal_stops_calling_the_lane_frozen() -> None:
    reason, detail = _classify({}, now=AFTER)
    assert reason is not None
    assert "frozen" not in detail


def test_a_compose_dev_pass_still_admits() -> None:
    reason, _detail = _classify({"compose-dev": "PASS"})
    assert reason is None


# --- the staging delivery gate ------------------------------------------------


def _gate_main(
    monkeypatch: pytest.MonkeyPatch, present: EnumLabLane | None, *extra: str
) -> int:
    from scripts.ci import lab_pass_receipt

    # The delivery's any-of premise is met by the onex-lab boot receipt the same
    # workflow emits; the ALL-OF required lane is what these tests vary.
    surface = FakeSurface()
    surface.add(receipt(SHA_4ACA, EnumLabLane.ONEX_LAB))
    if present is not None:
        surface.add(receipt(SHA_4ACA, present))
    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
    return lab_pass_receipt.main(
        ["gate", "--sha", SHA_4ACA, "--repo", "OmniNode-ai/omnibase_infra", *extra]
    )


def test_the_gate_requires_the_routed_lane_and_a_substitute_pass_admits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = ("--require-routed-lane", "omnibase_infra")
    assert _gate_main(monkeypatch, EnumLabLane.COMPOSE_DEV_202, *args) == 0


def test_the_gate_refuses_a_compose_dev_pass_when_the_route_is_the_substitute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = ("--require-routed-lane", "omnibase_infra")
    assert _gate_main(monkeypatch, EnumLabLane.COMPOSE_DEV, *args) != 0


def test_the_gate_refuses_with_no_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    args = ("--require-routed-lane", "omnibase_infra")
    assert _gate_main(monkeypatch, None, *args) != 0


def test_the_gate_refuses_a_substitute_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.ci import lab_pass_receipt

    surface = FakeSurface()
    surface.add(receipt(SHA_4ACA, EnumLabLane.ONEX_LAB))
    surface.add(receipt(SHA_4ACA, EnumLabLane.COMPOSE_DEV_202, outcome="fail"))
    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
    code = lab_pass_receipt.main(
        [
            "gate",
            "--sha",
            SHA_4ACA,
            "--repo",
            "OmniNode-ai/omnibase_infra",
            "--require-routed-lane",
            "omnibase_infra",
        ]
    )
    assert code != 0


def test_an_unrouted_repo_is_required_on_compose_dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = ("--require-routed-lane", "omnibase_core")
    assert _gate_main(monkeypatch, EnumLabLane.COMPOSE_DEV, *args) == 0
    assert _gate_main(monkeypatch, EnumLabLane.COMPOSE_DEV_202, *args) != 0


def test_an_unreadable_table_refuses_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.ci import lab_pass_receipt

    module = lab_pass_receipt._instance_receipt_lanes()
    monkeypatch.setattr(module, "ROUTING_TABLE", tmp_path / "absent.yaml")
    args = ("--require-routed-lane", "omnibase_infra")
    assert _gate_main(monkeypatch, EnumLabLane.COMPOSE_DEV_202, *args) == 1


def test_the_delivery_workflow_reads_the_routed_lane_not_a_literal() -> None:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    runs = [
        str(step["run"])
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Read the lab-pass receipt for the delivered sha"
    ]
    assert len(runs) == 1
    code = "\n".join(
        line for line in runs[0].splitlines() if not line.lstrip().startswith("#")
    )
    assert '--require-routed-lane "${GITHUB_REPOSITORY##*/}"' in code
    assert "--require-lane" not in code
    assert "compose-dev-202" not in code
    assert "--with pyyaml" in code
