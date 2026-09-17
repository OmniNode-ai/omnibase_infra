# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Derive the gateway deploy ceiling from the gateway's own model (OMN-18200).

WHY THIS EXISTS
----------------

``GATEWAY_DEPLOY_TIMEOUT_SECONDS`` was ``RUNTIME_IMAGE_BUILD_FLOOR_SECONDS +
300`` (900s) -- a floor plus a flat constant, unrelated to the gateway's own
build or reload cost. This is the same defect class OMN-18057 and OMN-18072
already closed for the runtime family: a number with no relationship to the
work it bounds.

MEASURED, from the deploy agent's own journal on the one rebuild since
``omnibase_infra#3524`` (``732fd291``) merged: ``_deploy_gateway_lane`` started
at ``05:15:22`` and ``bash scripts/deploy-gateway.sh --execute`` was killed at
``05:30:22`` -- exactly 900s. The built image ``docker-gateway-forwarder:build``
carries a creation time about thirteen minutes after the step began, so roughly
780 of the 900 seconds went to a cold image build, and the script was killed
during the recreate or the post-reload verify. ``status: failed`` /
``runtime: failed`` is CORRECT behaviour for a real failure; the defect is the
ceiling, not the outcome it produced.

THE MODEL
---------

``scripts/deploy-gateway.sh --execute`` does two separable things inside the
one subprocess this agent invokes:

1. Build. ``build_image`` + ``build_sidecar_image`` each run ONE ``docker
   compose ... build`` over ``docker/docker-compose.gateway.yml``:
   ``gateway-forwarder`` builds from the SAME ``docker/Dockerfile.runtime`` the
   runtime family builds (a strict subset -- one image, not nine), and
   ``gateway-dns-bastion`` builds from its own trivial ``FROM alpine:3.20``
   Dockerfile. This is exactly the model ``build_budget.derive_image_build_budget``
   already derives for the runtime family: reused here unchanged, over the
   gateway compose file instead of the runtime one.
2. Recreate. Resolve+retain the previous digest, sync two host files and diff
   them back, rewrite ``gateway.env``'s digest line, write ``registry.json``,
   reload (``systemctl reload`` -> the unit's own ``ExecReload``, which is
   ``docker compose ... up -d --force-recreate --wait --wait-timeout 120
   gateway-forwarder``), then ``verify_deployment`` (one ``docker inspect`` +
   two ``docker exec test -f`` calls, no loop). Every one of these is a single
   fast local command; only the reload half has a real, model-declared bound --
   the SAME ``--wait-timeout`` ``systemctl reload`` itself will enforce, read
   from the unit file the reload step actually invokes rather than duplicated
   as a second literal here.

    ceiling = build_ceiling + max(recreate_floor, wait_timeout + margin)

Both terms are READ from the same files the deploy is about to invoke (the
compose file's build specs and Dockerfiles, and the systemd unit's own
``ExecReload=`` line), so a change to either moves the ceiling with it instead
of silently re-opening this failure one number later -- the same property
``compose_budget`` and ``build_budget`` already hold for the runtime family.

THE FLOORS
----------

The build floor is ``RUNTIME_IMAGE_BUILD_FLOOR_SECONDS`` (600s, OMN-18072's own
floor) -- reused, not restated, because the gateway build runs the identical
BuildKit solve model over the same Dockerfile family. The recreate floor
guards against a unit file edited to declare a suspiciously small
``--wait-timeout``: the derived recreate ceiling can never fall at or below the
120s currently declared without a margin around it, because a ceiling that
silently collapsed to the raw wait-timeout would kill the reload the instant
compose's own internal retry/settle behaviour added even one second of
overhead outside the timed command itself.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from deploy_agent.build_budget import ModelBuildBudget, derive_image_build_budget
from deploy_agent.host_conditions import ModelHostConditions

logger = logging.getLogger(__name__)

# `ExecReload=... --wait-timeout 120 ...` -- accepts `--wait-timeout=120` too,
# though the unit today spells it as two tokens.
_WAIT_TIMEOUT_RE = re.compile(r"--wait-timeout[ =](\d+)")


class GatewayBudgetError(RuntimeError):
    """Raised when the gateway deploy model cannot be read to derive a ceiling.

    A ceiling that silently fell back to a floor because the model was
    unreadable is the same class of undetectable wrongness as the bare
    constant this module replaces (see ``BuildBudgetError`` /
    ``ComposeBudgetError``, the same fail-closed contract).
    """


class ModelGatewayDeployBudget(BaseModel):
    """A derived gateway deploy ceiling and the facts it was derived from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout_seconds: int
    build: ModelBuildBudget
    reload_wait_timeout_seconds: int
    reload_margin_seconds: int
    reload_floor_seconds: int
    recreate_seconds: int
    service_unit: str

    def describe(self) -> str:
        """One-line, log-ready statement of the ceiling and its source."""
        return (
            f"{self.timeout_seconds}s = build {self.build.timeout_seconds}s "
            f"({self.build.describe()}) + recreate {self.recreate_seconds}s "
            f"(ExecReload --wait-timeout {self.reload_wait_timeout_seconds}s "
            f"of {self.service_unit!r} + margin {self.reload_margin_seconds}s, "
            f"floor {self.reload_floor_seconds}s)"
        )


def read_exec_reload_wait_timeout(service_unit_path: str | Path) -> int:
    """Return the ``--wait-timeout`` seconds declared on the unit's ``ExecReload=`` line.

    Raises ``GatewayBudgetError`` when the unit cannot be read, declares no
    ``ExecReload=`` line, or that line declares no ``--wait-timeout`` -- the
    exact bound ``systemctl reload`` will enforce, and the recreate ceiling
    must never be derived without seeing it.
    """
    path = Path(service_unit_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GatewayBudgetError(
            f"cannot read systemd unit {path} to derive the gateway recreate "
            f"ceiling: {exc}"
        ) from exc
    exec_reload_line = next(
        (line for line in text.splitlines() if line.strip().startswith("ExecReload=")),
        None,
    )
    if exec_reload_line is None:
        raise GatewayBudgetError(
            f"{path} declares no ExecReload= line; the gateway reload step "
            "invokes this unit's ExecReload and the recreate ceiling cannot "
            "be derived without it"
        )
    match = _WAIT_TIMEOUT_RE.search(exec_reload_line)
    if match is None:
        raise GatewayBudgetError(
            f"{path}'s ExecReload= line declares no --wait-timeout; the "
            "recreate ceiling cannot be derived without the same bound "
            "'systemctl reload' will enforce"
        )
    return int(match.group(1))


def derive_gateway_deploy_budget(
    compose_files: tuple[str, ...] | list[str],
    profile: str,
    service_unit_path: str | Path,
    *,
    per_step_seconds: int,
    per_image_seconds: int,
    build_floor_seconds: int,
    reload_margin_seconds: int,
    reload_floor_seconds: int,
    host: ModelHostConditions | None = None,
) -> ModelGatewayDeployBudget:
    """Derive the ``scripts/deploy-gateway.sh --execute`` ceiling.

    Sums the build ceiling (``build_budget.derive_image_build_budget`` over
    the gateway compose file) and the recreate ceiling (the unit's own
    ``ExecReload --wait-timeout`` plus a margin for the surrounding local, non
    -polling work), so a cold build can never consume the budget the recreate
    half needs -- the defect the flat floor-plus-constant it replaces could
    not express.

    OMN-18615, SECOND PASS. ``host`` is threaded into the BUILD half and
    deliberately not into the recreate half. The build is work this ceiling
    actually bounds, and it slows down on a contended or cache-pruned machine
    exactly like the runtime build does. The recreate half is the unit's OWN
    ``ExecReload --wait-timeout``, which ``systemctl reload`` enforces itself --
    inflating it here would describe a bound nothing honours.

    Why this argument exists at all: the first pass wired the machine terms at
    ``executor.runtime_image_build_budget`` and MISSED this caller, so the
    gateway deploy kept deriving a ceiling as though the host were idle. On
    2026-09-17 that killed job ``a0b496ed`` at 1180s on a host whose load1 had
    peaked at 73.73. The kill message said so in its own words -- "no host
    conditions were read (model terms only ... blind to the machine)" -- which
    is the clause the first pass added for precisely this case.

    Raises ``GatewayBudgetError`` when either half's model cannot be read.
    """
    build = derive_image_build_budget(
        compose_files,
        profile,
        per_step_seconds=per_step_seconds,
        per_image_seconds=per_image_seconds,
        floor_seconds=build_floor_seconds,
        host=host,
    )
    reload_wait_timeout = read_exec_reload_wait_timeout(service_unit_path)
    recreate_seconds = max(
        reload_floor_seconds, reload_wait_timeout + reload_margin_seconds
    )
    return ModelGatewayDeployBudget(
        timeout_seconds=build.timeout_seconds + recreate_seconds,
        build=build,
        reload_wait_timeout_seconds=reload_wait_timeout,
        reload_margin_seconds=reload_margin_seconds,
        reload_floor_seconds=reload_floor_seconds,
        recreate_seconds=recreate_seconds,
        service_unit=str(service_unit_path),
    )
