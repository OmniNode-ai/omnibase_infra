# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19507 AC2: the per-merge verify job reads the lane the merge was routed to.

Task B5 of the second-deploy-slot plan (epic OMN-19500). The deploy agent routes
each dev rebuild command to one instance by ``config/deploy_lane_routing.yaml``
(OMN-19506). The verify job that polls convergence and emits the sha's lab-pass
receipt now takes its runner, its receipt lane and every probe target from the
routed instance's ``verify:`` block, through
``scripts/ci/deploy_lane_verify_route.py``.

THE CONTROL THAT MATTERS MOST: the committed table routes nothing to dev-202, so
every merge in both workflows still resolves to dev-201, and dev-201's block is
literally the values the two workflows hard-coded before this change. So the
behaviour on .201 is unchanged, and that is asserted below rather than argued.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci import check_dev_lane_staleness, check_lane_sibling_revision
from scripts.ci.deploy_lane_verify_route import (
    ROUTING_TABLE,
    VerifyRouteError,
    job_env,
    job_outputs,
    load_table,
    resolve,
    route,
    targets_for_receipt_lane,
)
from scripts.ci.lab_pass_receipt import ANY_OF_DEFAULT_LANES, EnumLabLane
from scripts.ci.lane_settle_budget import (
    assert_declaration_within_bounds,
    load_declaration,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
DIRECT = WORKFLOWS / "runtime-rebuild-trigger.yml"
REUSABLE = WORKFLOWS / "runtime-rebuild-trigger-reusable.yml"
DEV_202_OVERLAY = REPO_ROOT / "docker" / "docker-compose.dev-202.yml"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "deploy-agent"))
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.routing import parse_routing_table

#: What runtime-rebuild-trigger.yml and its reusable hard-coded before OMN-19507.
PRE_CHANGE_201 = {
    "receipt_lane": "compose-dev",
    "runner_labels": ("self-hosted", "omnibase-verify", "host-201"),
    "deploy_agent_url": "http://host.docker.internal:8098",
    "main_url": "http://host.docker.internal:8085",
    "effects_url": "http://host.docker.internal:8086",
    "projection_url": "http://host.docker.internal:3002",
    "compose_project": "omnibase-infra",
    "runtime_container": "omninode-runtime",
    "effects_container": "omninode-runtime-effects",
    "postgres_container": "omnibase-infra-postgres",
    "broker_container": "omnibase-infra-redpanda",
}

REQUESTERS = (
    "gha/omnibase_infra/pr-4099",
    "gha/omnimarket/pr-2870",
    "gha/omniclaude/pr-1",
    "operator/jonah",
    "agent/idle-converge",
)


def _with_omnimarket_row() -> dict[str, Any]:
    """The plan's target table: (dev, omnimarket) -> dev-202 (task B8)."""
    table = copy.deepcopy(load_table())
    table["routes"] = [
        {
            "runtime_lane": "dev",
            "requester_repository": "omnimarket",
            "instance": "dev-202",
        }
    ]
    return table


# --------------------------------------------------------------------------- #
# .201 is unchanged                                                            #
# --------------------------------------------------------------------------- #
class TestTheCommittedTableKeepsEveryMergeOn201:
    @pytest.mark.parametrize("requested_by", REQUESTERS)
    def test_every_requester_resolves_to_dev_201_with_the_pre_change_values(
        self, requested_by: str
    ) -> None:
        resolved = resolve(load_table(), runtime_lane="dev", requested_by=requested_by)
        assert resolved.instance == "dev-201"
        assert resolved.targets.model_dump() == PRE_CHANGE_201

    def test_the_job_outputs_are_the_old_runner_and_lane(self) -> None:
        outputs = job_outputs(
            resolve(
                load_table(),
                runtime_lane="dev",
                requested_by="gha/omnibase_infra/pr-1",
            )
        )
        assert outputs == {
            "routed_instance": "dev-201",
            "receipt_lane": "compose-dev",
            "verify_runs_on": '["self-hosted","omnibase-verify","host-201"]',
        }
        assert json.loads(outputs["verify_runs_on"]) == list(
            PRE_CHANGE_201["runner_labels"]
        )

    def test_dev_201_agrees_with_the_guards_own_defaults(self) -> None:
        """The two convergence guards default to the .201 lane; its declared
        block names the same containers and project."""
        targets = targets_for_receipt_lane(load_table(), "compose-dev")
        assert targets.runtime_container == check_dev_lane_staleness.DEV_LANE_CONTAINER
        assert (
            targets.compose_project == check_dev_lane_staleness.DEV_LANE_COMPOSE_PROJECT
        )
        assert (
            targets.effects_container == check_lane_sibling_revision.DEFAULT_CONTAINER
        )
        assert (
            targets.compose_project
            == check_lane_sibling_revision.DEFAULT_COMPOSE_PROJECT
        )

    def test_omnibase_infra_is_never_routed_off_201(self) -> None:
        """Operator ruling 2026-09-25T00:56:45Z: omnibase_infra changes stay
        proven on .201 only."""
        table = load_table()
        for row in table.get("routes") or ():
            if row.get("requester_repository") == "omnibase_infra":
                assert row.get("instance") == "dev-201", row


# --------------------------------------------------------------------------- #
# dev-202, once the route row exists                                           #
# --------------------------------------------------------------------------- #
class TestADev202RouteMovesOnlyOmnimarket:
    def test_omnimarket_resolves_to_dev_202_and_its_own_receipt_lane(self) -> None:
        resolved = resolve(
            _with_omnimarket_row(),
            runtime_lane="dev",
            requested_by="gha/omnimarket/pr-2870",
        )
        assert resolved.instance == "dev-202"
        assert resolved.targets.receipt_lane == "compose-dev-202"
        assert json.loads(job_outputs(resolved)["verify_runs_on"]) == [
            "self-hosted",
            "omnibase-verify",
            "host-202",
        ]

    def test_omnibase_infra_still_resolves_to_dev_201(self) -> None:
        resolved = resolve(
            _with_omnimarket_row(),
            runtime_lane="dev",
            requested_by="gha/omnibase_infra/pr-1",
        )
        assert resolved.instance == "dev-201"
        assert resolved.targets.model_dump() == PRE_CHANGE_201

    @pytest.mark.parametrize("table_name", ["committed", "with_omnimarket_row"])
    @pytest.mark.parametrize("requested_by", REQUESTERS)
    def test_the_verify_route_is_the_deploy_agents_route(
        self, table_name: str, requested_by: str
    ) -> None:
        """One rule, stated twice: the verify job must read the lane the agent
        actually ran the command on."""
        table = load_table() if table_name == "committed" else _with_omnimarket_row()
        agent_table = parse_routing_table(yaml.safe_dump(table))
        assert route(
            table,
            "dev",
            check_dev_lane_staleness_requester(requested_by),
        ) == agent_table.route(EnumRuntimeLane.DEV, requested_by)


def check_dev_lane_staleness_requester(requested_by: str) -> str | None:
    from scripts.ci.deploy_lane_verify_route import requester_repository

    return requester_repository(requested_by)


# --------------------------------------------------------------------------- #
# The declarations                                                             #
# --------------------------------------------------------------------------- #
class TestEveryInstanceDeclaresHowItIsVerified:
    def test_each_instance_has_a_distinct_known_receipt_lane(self) -> None:
        table = load_table()
        lanes = [spec["verify"]["receipt_lane"] for spec in table["instances"].values()]
        assert len(lanes) == len(set(lanes))
        for lane in lanes:
            EnumLabLane(lane)

    def test_dev_202s_lane_stays_out_of_the_any_of_default(self) -> None:
        targets = targets_for_receipt_lane(load_table(), "compose-dev-202")
        assert EnumLabLane(targets.receipt_lane) not in ANY_OF_DEFAULT_LANES

    def test_each_runner_is_host_scoped_and_never_the_customer_runner(self) -> None:
        table = load_table()
        for name, spec in table["instances"].items():
            labels = spec["verify"]["runner_labels"]
            assert labels[:2] == ["self-hosted", "omnibase-verify"], name
            assert f"host-{name.split('-')[-1]}" in labels, name
            assert "omnipc2-customer" not in labels, name

    def test_dev_202_targets_agree_with_its_overlay(self) -> None:
        targets = targets_for_receipt_lane(load_table(), "compose-dev-202")
        overlay_text = DEV_202_OVERLAY.read_text(encoding="utf-8")
        overlay = yaml.load(overlay_text, Loader=_OverrideLoader)  # noqa: S506
        services = overlay["services"]
        assert overlay["name"] == targets.compose_project
        assert services["omninode-runtime"]["container_name"] == (
            targets.runtime_container
        )
        assert services["runtime-effects"]["container_name"] == (
            targets.effects_container
        )
        assert services["postgres"]["container_name"] == targets.postgres_container
        assert services["redpanda"]["container_name"] == targets.broker_container
        for url in (targets.main_url, targets.effects_url, targets.projection_url):
            port = re.search(r":(\d+)$", url)
            assert port is not None
            assert re.search(rf":{port.group(1)}:\d+", overlay_text), url

    @pytest.mark.parametrize("lane", ["compose-dev", "compose-dev-202"])
    def test_each_verified_lane_has_a_settle_budget_inside_its_bounds(
        self, lane: str
    ) -> None:
        declaration = load_declaration(lane)
        assert_declaration_within_bounds(declaration)


class _OverrideLoader(yaml.SafeLoader):
    """Compose's ``!override`` and ``!reset`` tags resolve to their value."""


_OverrideLoader.add_constructor(
    "!override",
    lambda loader, node: (
        loader.construct_sequence(node)
        if isinstance(node, yaml.SequenceNode)
        else loader.construct_mapping(node)
        if isinstance(node, yaml.MappingNode)
        else loader.construct_scalar(node)
    ),
)
_OverrideLoader.add_constructor("!reset", lambda loader, node: None)


# --------------------------------------------------------------------------- #
# Fail-closed                                                                  #
# --------------------------------------------------------------------------- #
class TestTheResolverRefusesRatherThanGuesses:
    def test_an_instance_with_no_verify_block_refuses(self) -> None:
        table = _with_omnimarket_row()
        del table["instances"]["dev-202"]["verify"]
        with pytest.raises(VerifyRouteError, match="no verify"):
            resolve(table, runtime_lane="dev", requested_by="gha/omnimarket/pr-1")

    def test_a_missing_field_refuses(self) -> None:
        table = load_table()
        del table["instances"]["dev-201"]["verify"]["runner_labels"]
        with pytest.raises(VerifyRouteError, match="invalid"):
            resolve(table, runtime_lane="dev", requested_by="gha/omnibase_infra/x")

    def test_a_receipt_lane_no_instance_emits_refuses(self) -> None:
        with pytest.raises(VerifyRouteError, match="0 instances"):
            targets_for_receipt_lane(load_table(), "onex-lab")

    def test_the_cli_writes_the_outputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "out"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))
        assert (
            check_dev_lane_staleness.main(
                [
                    "--write-route-outputs",
                    "--runtime-lane",
                    "dev",
                    "--requested-by",
                    "gha/omnibase_infra/pr-7",
                ]
            )
            == 0
        )
        assert out.read_text().splitlines() == [
            "routed_instance=dev-201",
            "receipt_lane=compose-dev",
            'verify_runs_on=["self-hosted","omnibase-verify","host-201"]',
        ]

    def test_the_cli_writes_the_lane_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = tmp_path / "env"
        monkeypatch.setenv("GITHUB_ENV", str(env))
        assert (
            check_dev_lane_staleness.main(["--write-lane-env", "--lane", "compose-dev"])
            == 0
        )
        written = dict(line.split("=", 1) for line in env.read_text().splitlines())
        assert written == job_env(targets_for_receipt_lane(load_table(), "compose-dev"))
        assert written["DEPLOY_AGENT_URL"] == PRE_CHANGE_201["deploy_agent_url"]

    def test_the_cli_writes_nothing_when_it_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = tmp_path / "env"
        monkeypatch.setenv("GITHUB_ENV", str(env))
        assert (
            check_dev_lane_staleness.main(["--write-lane-env", "--lane", "onex-lab"])
            == 1
        )
        assert not env.exists()

    def test_the_committed_table_is_the_default(self) -> None:
        assert ROUTING_TABLE == REPO_ROOT / "config" / "deploy_lane_routing.yaml"


# --------------------------------------------------------------------------- #
# check_dev_lane_staleness.py --lane                                           #
# --------------------------------------------------------------------------- #
class TestTheStalenessGuardTakesALane:
    def test_no_lane_keeps_the_201_defaults(self) -> None:
        assert check_dev_lane_staleness.resolve_lane_target("", None, None) == (
            "omninode-runtime",
            "omnibase-infra",
        )

    def test_compose_dev_is_the_201_lane(self) -> None:
        assert check_dev_lane_staleness.resolve_lane_target(
            "compose-dev", None, None
        ) == ("omninode-runtime", "omnibase-infra")

    def test_compose_dev_202_reads_the_202_lane(self) -> None:
        assert check_dev_lane_staleness.resolve_lane_target(
            "compose-dev-202", None, None
        ) == ("omninode-dev-202-runtime", "omnibase-infra-dev-202")

    def test_an_explicit_target_that_contradicts_the_lane_refuses(self) -> None:
        with pytest.raises(ValueError, match="compose-dev-202"):
            check_dev_lane_staleness.resolve_lane_target(
                "compose-dev-202", "omninode-runtime", None
            )

    def test_the_cli_fails_closed_on_an_unknown_lane(self) -> None:
        assert (
            check_dev_lane_staleness.main(
                ["--lane", "onex-lab", "--deployed-revision", "a" * 40]
            )
            == 1
        )


# --------------------------------------------------------------------------- #
# The workflows                                                                #
# --------------------------------------------------------------------------- #
def _jobs(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]


def _run_text(job: dict[str, Any]) -> str:
    return "\n".join(str(step.get("run", "")) for step in job["steps"])


@pytest.mark.parametrize(
    ("path", "verify_job", "requester"),
    [
        (DIRECT, "verify-lane-converged", "gha/omnibase_infra/pr-$PR_NUMBER"),
        (
            REUSABLE,
            "verify-sibling-converged",
            "gha/${SOURCE_REPO}/pr-${PR_NUMBER}",
        ),
    ],
    ids=["direct", "reusable"],
)
class TestTheWorkflowsReadTheRoute:
    def test_the_trigger_job_resolves_the_route_with_the_published_requester(
        self, path: Path, verify_job: str, requester: str
    ) -> None:
        trigger = _jobs(path)["trigger-rebuild"]
        text = _run_text(trigger)
        assert "--write-route-outputs" in text
        assert "scripts/ci/check_dev_lane_staleness.py" in text
        assert f'--requested-by "{requester}"' in text
        for name in ("routed_instance", "receipt_lane", "verify_runs_on"):
            assert name in trigger["outputs"], name

    def test_the_verify_job_runs_where_the_route_says(
        self, path: Path, verify_job: str, requester: str
    ) -> None:
        job = _jobs(path)[verify_job]
        assert job["runs-on"] == (
            "${{ fromJSON(needs.trigger-rebuild.outputs.verify_runs_on) }}"
        )

    def test_the_verify_job_takes_its_lane_env_from_the_route(
        self, path: Path, verify_job: str, requester: str
    ) -> None:
        job = _jobs(path)[verify_job]
        text = _run_text(job)
        assert "--write-lane-env" in text
        assert '--lane "$ROUTED_RECEIPT_LANE"' in text
        assert "--lane compose-dev " not in text + " "
        assert "--lane compose-dev\n" not in text + "\n"
        assert "host.docker.internal" not in yaml.safe_dump(job["steps"])
        upload = [s for s in job["steps"] if "upload-artifact" in str(s.get("uses"))]
        assert upload
        for step in upload:
            assert "outputs.receipt_lane" in step["with"]["name"]
