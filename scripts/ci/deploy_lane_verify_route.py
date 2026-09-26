# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which dev-lane instance a merge's rebuild runs on, and how its verify job reads it.

OMN-19507 AC2, task B5 of the second-deploy-slot plan (knowledge-base-internal
``beta/plans/2026-09-24-second-deploy-slot-plan.md``, epic OMN-19500).

WHY THIS EXISTS
---------------
The deploy agent routes each dev rebuild command to ONE instance by the declared
table ``config/deploy_lane_routing.yaml`` (OMN-19506): dev-201 on .201, or dev-202
on .202. The verify job that waits for the lane to converge and then emits the
sha's lab-pass receipt read only the .201 lane: it was pinned to a ``host-201``
runner, polled the .201 deploy agent and containers, and wrote the
``compose-dev`` receipt. A merge routed to dev-202 would have been "verified"
against a lane that never ran it.

So the verify job now takes everything lane-specific from the instance the
table routes the merge to, declared in that instance's ``verify:`` block:

* ``receipt_lane``: the lab-pass lane the receipt is emitted under. Each lane
  name has one emitter (``lab_pass_receipt.EnumLabLane``), so dev-202 proves
  under ``compose-dev-202`` and never under ``compose-dev``.
* ``runner_labels``: the self-hosted runner that can see that host's docker
  daemon.
* the deploy agent's HTTP surface, the lane's three readiness URLs, and the
  containers the guards inspect.

THE ROUTE IS THE AGENT'S ROUTE
------------------------------
:func:`route` is the same rule as ``deploy_agent.routing.ModelRoutingTable.route``
(a ``(runtime_lane, requester_repository)`` row, else the default), and
``tests/ci/test_deploy_lane_verify_route_omn19507.py`` checks the two agree on
the committed table and on a table with a dev-202 row. It is re-stated rather
than imported because the deploy agent is a separate package with its own
environment, and this runs in the CI one.

FAIL-CLOSED
-----------
An unreadable table, an instance with no ``verify:`` block, a missing field, or
a routed instance the table does not declare each raise
:class:`VerifyRouteError`. The workflow entry points (``lab_pass_receipt.py
route-outputs`` and ``route-env``, which call :func:`write_route_outputs` and
:func:`write_lane_env`) then exit non-zero and write nothing, so the job's
outputs stay empty and the verify job cannot start on a guessed runner.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
ROUTING_TABLE: Final = _REPO_ROOT / "config" / "deploy_lane_routing.yaml"

#: ``requested_by`` of a CI-published command: ``gha/<repo>/...``.
_CI_REQUESTER_RE: Final = re.compile(r"^gha/([a-z][a-z0-9_]*)/")


class VerifyRouteError(RuntimeError):
    """The table cannot say which instance, or how to verify it."""


class ModelVerifyTargets(BaseModel):
    """One instance's ``verify:`` block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    receipt_lane: str
    runner_labels: tuple[str, ...]
    deploy_agent_url: str
    main_url: str
    effects_url: str
    projection_url: str
    compose_project: str
    runtime_container: str
    effects_container: str
    postgres_container: str
    broker_container: str


class ModelVerifyRoute(BaseModel):
    """The routed instance and its verify targets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instance: str
    targets: ModelVerifyTargets


def requester_repository(requested_by: str) -> str | None:
    """``gha/<repo>/...`` -> ``<repo>``; any other requester -> ``None``."""
    match = _CI_REQUESTER_RE.match(requested_by)
    return match.group(1) if match else None


def load_table(path: Path = ROUTING_TABLE) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise VerifyRouteError(f"routing table unreadable at {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise VerifyRouteError(f"routing table at {path} is not a mapping")
    return raw


def route(table: dict[str, Any], runtime_lane: str, repository: str | None) -> str:
    """The instance that runs a ``runtime_lane`` command from ``repository``."""
    default = str(table.get("default_instance") or "").strip()
    if not default:
        raise VerifyRouteError("routing table declares no default_instance")
    for row in table.get("routes") or ():
        if (
            isinstance(row, dict)
            and str(row.get("runtime_lane")) == runtime_lane
            and str(row.get("requester_repository")) == repository
        ):
            return str(row.get("instance"))
    return default


def verify_targets(table: dict[str, Any], instance: str) -> ModelVerifyTargets:
    instances = table.get("instances")
    if not isinstance(instances, dict) or instance not in instances:
        raise VerifyRouteError(f"instance {instance!r} is not declared")
    spec = instances[instance]
    block = spec.get("verify") if isinstance(spec, dict) else None
    if not isinstance(block, dict):
        raise VerifyRouteError(f"instance {instance!r} declares no verify: block")
    try:
        return ModelVerifyTargets.model_validate(block)
    except ValidationError as exc:
        raise VerifyRouteError(
            f"instance {instance!r} verify: block is invalid: {exc}"
        ) from exc


def resolve(
    table: dict[str, Any], *, runtime_lane: str, requested_by: str
) -> ModelVerifyRoute:
    instance = route(table, runtime_lane, requester_repository(requested_by))
    return ModelVerifyRoute(instance=instance, targets=verify_targets(table, instance))


def targets_for_receipt_lane(
    table: dict[str, Any], receipt_lane: str
) -> ModelVerifyTargets:
    """The verify targets of the one instance that emits ``receipt_lane``."""
    instances = table.get("instances")
    if not isinstance(instances, dict):
        raise VerifyRouteError("routing table declares no instances")
    found = [
        verify_targets(table, str(name))
        for name, spec in instances.items()
        if isinstance(spec, dict)
        and isinstance(spec.get("verify"), dict)
        and spec["verify"].get("receipt_lane") == receipt_lane
    ]
    if len(found) != 1:
        raise VerifyRouteError(
            f"{len(found)} instances emit receipt lane {receipt_lane!r}; exactly "
            "one must (a lab-pass lane name has one emitter)"
        )
    return found[0]


def job_outputs(resolved: ModelVerifyRoute) -> dict[str, str]:
    """The trigger job's outputs the verify job's ``runs-on`` and names read."""
    return {
        "routed_instance": resolved.instance,
        "receipt_lane": resolved.targets.receipt_lane,
        "verify_runs_on": json.dumps(
            list(resolved.targets.runner_labels), separators=(",", ":")
        ),
    }


def job_env(targets: ModelVerifyTargets) -> dict[str, str]:
    """The verify job's lane environment, written once to ``$GITHUB_ENV``."""
    return {
        "RECEIPT_LANE": targets.receipt_lane,
        "DEPLOY_AGENT_URL": targets.deploy_agent_url,
        "DEV_LANE_MAIN_URL": targets.main_url,
        "DEV_LANE_EFFECTS_URL": targets.effects_url,
        "DEV_LANE_PROJECTION_URL": targets.projection_url,
        "DEV_LANE_COMPOSE_PROJECT": targets.compose_project,
        "DEV_LANE_RUNTIME_CONTAINER": targets.runtime_container,
        "DEV_LANE_EFFECTS_CONTAINER": targets.effects_container,
        "DEV_LANE_POSTGRES_CONTAINER": targets.postgres_container,
        "DEV_LANE_BROKER_CONTAINER": targets.broker_container,
    }


def _append(path_var: str, values: dict[str, str]) -> None:
    target = os.environ.get(path_var, "")
    lines = "".join(f"{key}={value}\n" for key, value in values.items())
    if target:
        with Path(target).open("a", encoding="utf-8") as handle:
            handle.write(lines)
    sys.stdout.write(lines)


def write_route_outputs(
    *, runtime_lane: str, requested_by: str, table_path: Path = ROUTING_TABLE
) -> int:
    """``lab_pass_receipt.py route-outputs``: the trigger job's outputs."""
    try:
        resolved = resolve(
            load_table(table_path),
            runtime_lane=runtime_lane,
            requested_by=requested_by,
        )
    except VerifyRouteError as exc:
        print(f"::error::deploy-lane verify route: {exc}", file=sys.stderr)
        return 1
    _append("GITHUB_OUTPUT", job_outputs(resolved))
    return 0


def write_lane_env(*, receipt_lane: str, table_path: Path = ROUTING_TABLE) -> int:
    """``lab_pass_receipt.py route-env``: one receipt lane's verify environment."""
    try:
        targets = targets_for_receipt_lane(load_table(table_path), receipt_lane)
    except VerifyRouteError as exc:
        print(f"::error::deploy-lane verify route: {exc}", file=sys.stderr)
        return 1
    _append("GITHUB_ENV", job_env(targets))
    return 0
