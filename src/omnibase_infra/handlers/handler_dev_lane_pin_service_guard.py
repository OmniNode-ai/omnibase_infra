# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A code-ref pin cannot send the build a service the compose lacks (OMN-23368).

THE FAILURE
-----------
The deploy agent's service lists are part of the code a code-ref pin (OMN-23039) loads;
the compose files its build runs against come from the deploy clone. A pin on a branch
cut before a dev merge that removed a service therefore hands ``docker compose`` a name
the compose no longer declares. The dev deploy died that way twice on 2026-11-09
(``intelligence-api``, then ``infra-routing-decisions-consumer``), each time inside
docker with ``no such service`` and after the agent had begun work.

WHAT THIS DOES
--------------
Two readings, both taken before any build, by :func:`guard_before_build`:

* the pin's distance from the tracking ref: commits of ``origin/<tracking ref>`` that the
  agent's running code lacks. A pin behind dev is REPORTED (logged, and returned for the
  job's record); it never refuses a deploy on its own.
* the services the deploy is about to build and recreate against the services the lane's
  compose files declare. A name the compose lacks raises
  :class:`ComposeServiceMissingError`, naming the pin and every missing service.

Every dependency is injected: the compose reader, the process environment, the clock.
The standalone agent loads this module by file path (``load_dev_boot_guard``).
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_dev_lane_pin_staleness import (
    STALE_PIN_FRICTION,
    ModelDevLanePinStaleness,
)

if TYPE_CHECKING:
    from omnibase_infra.models.model_dev_lane_code_ref_pin import (
        ModelDevLaneCodeRefPin,
    )
    from omnibase_infra.models.model_dev_lane_pin_guard_request import (
        ModelDevLanePinGuardRequest,
    )
    from omnibase_infra.models.model_dev_lane_pin_guard_result import (
        ModelDevLanePinGuardResult,
    )

logger = logging.getLogger(__name__)

ENV_TRACKING_REF = "DEPLOY_AGENT_TRACKING_REF"
#: The agent's state directory, where the code-ref pin record lives (OMN-23039).
ENV_STATE_DIR = "DEPLOY_AGENT_STATE_DIR"
UNREADABLE_FRICTION = "code_ref_pin_staleness_unreadable"


class ComposeServiceMissingError(RuntimeError):
    """The deploy's service list names a service the lane's compose files lack."""

    def __init__(
        self,
        message: str,
        *,
        missing: tuple[str, ...],
        pin_label: str,
        compose_files: tuple[str, ...],
    ) -> None:
        super().__init__(message)
        self.missing = missing
        self.pin_label = pin_label
        self.compose_files = compose_files


def pin_label(
    env: Mapping[str, str],
    now: datetime,
    read_pin: Callable[[Path, datetime], ModelDevLaneCodeRefPin | None],
) -> str:
    """Name what the agent's code follows: the live code-ref pin, else the tracking ref."""
    state_dir = env.get(ENV_STATE_DIR, "").strip()
    pin = read_pin(Path(state_dir), now) if state_dir else None
    if pin is not None:
        return (
            f"code-ref pin {pin.branch} (pr={pin.pr} ticket={pin.ticket} "
            f"registered_head={pin.head[:12]})"
        )
    return f"tracking ref {env.get(ENV_TRACKING_REF, '').strip() or '<undeclared>'}"


def measure_pin_staleness(
    agent_dir: str, label: str, tracking_ref: str, *, git_timeout: float
) -> ModelDevLanePinStaleness | None:
    """Commits of ``origin/<tracking_ref>`` the clone's HEAD does not contain.

    ``None`` when git cannot answer; that is journalled, never read as "not behind".
    """
    dev_ref = f"origin/{tracking_ref}"
    result = subprocess.run(
        ["git", "-C", agent_dir, "rev-list", "--count", f"HEAD..{dev_ref}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=git_timeout,
    )
    text = result.stdout.strip()
    if result.returncode != 0 or not text.isdigit():
        logger.warning(
            "cannot count commits of %s missing from %s (exit=%s): %s friction_type=%s",
            dev_ref,
            label,
            result.returncode,
            result.stderr.strip()[:200],
            UNREADABLE_FRICTION,
        )
        return None
    return ModelDevLanePinStaleness(
        pin_label=label, dev_ref=dev_ref, behind_by=int(text)
    )


def assert_services_declared(
    services: Iterable[str],
    *,
    declared: Iterable[str],
    compose_files: Sequence[str],
    label: str,
    staleness: ModelDevLanePinStaleness | None = None,
) -> None:
    """Raise when ``services`` names one the compose files do not declare."""
    known = set(declared)
    missing = tuple(sorted({s for s in services if s not in known}))
    if not missing:
        return
    stale_note = (
        f" The pin is {staleness.behind_by} commit"
        f"{'' if staleness.behind_by == 1 else 's'} behind {staleness.dev_ref}."
        if staleness is not None and staleness.behind_by > 0
        else ""
    )
    raise ComposeServiceMissingError(
        f"deploy refused before any build: the service list of {label} names "
        f"{', '.join(missing)}, which the compose files ({', '.join(compose_files)}) "
        f"do not declare.{stale_note} Re-pin the agent onto a branch that carries the "
        f"dev compose, or release the pin.",
        missing=missing,
        pin_label=label,
        compose_files=tuple(compose_files),
    )


def guard_before_build(
    targets: Iterable[str],
    compose_files: Sequence[str],
    *,
    env: Mapping[str, str],
    agent_dir: str,
    read_services: Callable[[Sequence[str]], Mapping[str, Any]],
    read_pin: Callable[[Path, datetime], ModelDevLaneCodeRefPin | None],
    git_timeout: float,
    clock: Callable[[], datetime] | None = None,
) -> str:
    """Report a pin behind dev, then refuse a service the compose lacks.

    Returns the staleness report ("" when the agent's code is not behind). Raises
    :class:`ComposeServiceMissingError` before any docker call.
    """
    label = pin_label(env, (clock or (lambda: datetime.now(UTC)))(), read_pin)
    tracking_ref = env.get(ENV_TRACKING_REF, "").strip()
    staleness = (
        measure_pin_staleness(agent_dir, label, tracking_ref, git_timeout=git_timeout)
        if tracking_ref
        else None
    )
    report = staleness.describe() if staleness and staleness.behind_by > 0 else ""
    if report:
        logger.warning("%s", report)
    assert_services_declared(
        targets,
        declared=read_services(compose_files),
        compose_files=compose_files,
        label=label,
        staleness=staleness,
    )
    return report


class HandlerDevLanePinServiceGuard:
    """The node's single entry point; the deploy agent calls :func:`guard_before_build`."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelDevLanePinGuardRequest
    ) -> ModelDevLanePinGuardResult:
        """Report a pin behind dev, then refuse a service the compose lacks."""
        # The standalone deploy agent loads this module without the request and result
        # models (it only calls guard_before_build); the node path has them.
        from omnibase_infra.models.model_dev_lane_pin_guard_result import (
            ModelDevLanePinGuardResult,
        )

        report = guard_before_build(
            request.targets,
            request.compose_files,
            env=request.env,
            agent_dir=request.agent_dir,
            read_services=lambda _files: dict.fromkeys(request.declared_services),
            read_pin=lambda _dir, _now: request.pin,
            git_timeout=request.git_timeout_seconds,
            clock=lambda: request.now,
        )
        return ModelDevLanePinGuardResult(report=report)
