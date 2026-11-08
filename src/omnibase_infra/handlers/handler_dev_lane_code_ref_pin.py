# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A declared branch pin: the branch whose code the dev deploy agent loads (OMN-23039).

THE FAILURE
-----------
``self_update`` rebuilds the agent's code clone onto ``origin/<DEPLOY_AGENT_TRACKING_REF>``
at every idle heartbeat. A fix to the agent's own code (the boot guard, the executor)
could therefore not be shown working on the dev lane before its merge: put the clone
on the fix branch and the next heartbeat put it back on ``dev``. omnibase_infra#5772
was refused hand-off for that reason: no row could record the branch installed on the
h201 agent.

WHAT THIS IS
------------
A pin is a record in the agent's state directory (``ModelDevLaneCodeRefPin``) naming the
branch whose code the agent loads, with an expiry of at most seven days. ``self_update``
asks :func:`followed_branch` which branch to follow and, while a record is live, follows
the pinned one through the same fetch, rebuild and re-exec it uses for the tracking ref.
A released or expired record returns the agent to the tracking ref at its next heartbeat.

The pin never touches ``DEPLOY_AGENT_TRACKING_REF``: that still names the sibling
repository fallback of a build and the ref fence, and a sibling repository has no
branch of the pinned name. An unreadable or invalid record is ignored and journalled
(``friction_type=code_ref_pin_unreadable``): a record the agent cannot read must not
move the code it loads.

DEV ONLY. The agent reads the pin from its own state directory; the prod unit declares
its own and nothing registers a pin there. NO DEFAULTS (rule 8): the state directory is
always named by the caller.

Command line::

    python -m omnibase_infra.nodes.node_dev_lane_code_ref_pin_effect.runtime_code_ref_pin register \\
        --state-dir D --branch B --head SHA --pr N --ticket OMN-n \\
        --registered-by LANE --expires-in-hours 48
    python -m omnibase_infra.nodes.node_dev_lane_code_ref_pin_effect.runtime_code_ref_pin show --state-dir D
    python -m omnibase_infra.nodes.node_dev_lane_code_ref_pin_effect.runtime_code_ref_pin release --state-dir D
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_dev_lane_code_ref_pin import ModelDevLaneCodeRefPin

if TYPE_CHECKING:
    from omnibase_infra.models.model_dev_lane_code_ref_pin_request import (
        ModelDevLaneCodeRefPinRequest,
    )
    from omnibase_infra.models.model_dev_lane_code_ref_pin_result import (
        ModelDevLaneCodeRefPinResult,
    )

logger = logging.getLogger(__name__)

ENV_STATE_DIR = "DEPLOY_AGENT_STATE_DIR"
PIN_FILE_NAME = "code-ref-pin.json"


def pin_path(state_dir: Path) -> Path:
    """Where the record of ``state_dir`` lives."""
    return state_dir / PIN_FILE_NAME


def write_pin(state_dir: Path, pin: ModelDevLaneCodeRefPin) -> Path:
    """Write the record atomically; it replaces any earlier one."""
    target = pin_path(state_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".code-ref-pin.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(pin.model_dump_json(indent=2) + "\n")
        Path(tmp).replace(target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def release_pin(state_dir: Path) -> Path:
    """Drop the record; the agent follows its tracking ref at the next heartbeat."""
    target = pin_path(state_dir)
    target.unlink(missing_ok=True)
    return target


def read_active_pin(state_dir: Path, now: datetime) -> ModelDevLaneCodeRefPin | None:
    """The live record, or None when absent, expired or unreadable."""
    target = pin_path(state_dir)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning(
            "code-ref pin %s unreadable, ignored: %s "
            "friction_type=code_ref_pin_unreadable",
            target,
            exc,
        )
        return None
    try:
        pin = ModelDevLaneCodeRefPin.model_validate_json(text)
    except ValidationError as exc:
        logger.warning(
            "code-ref pin %s invalid, ignored: %s "
            "friction_type=code_ref_pin_unreadable",
            target,
            exc.errors(include_url=False, include_input=False),
        )
        return None
    if pin.expires_at <= now:
        logger.info(
            "code-ref pin for %s expired; following the tracking ref", pin.branch
        )
        return None
    return pin


class HandlerDevLaneCodeRefPin:
    """Register, read and release the code-ref pin of one agent state directory."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelDevLaneCodeRefPinRequest
    ) -> ModelDevLaneCodeRefPinResult:
        """Run the requested operation; the node's single entry point."""
        # The standalone deploy agent loads this module without the request and
        # result models (it only calls followed_branch); the node path has them.
        from omnibase_infra.models.model_dev_lane_code_ref_pin_result import (
            ModelDevLaneCodeRefPinResult,
        )

        if request.operation == "register":
            if request.pin is None:  # unreachable past the request validator
                raise ValueError("register needs a pin")
            path = self.register(request.state_dir, request.pin)
            return ModelDevLaneCodeRefPinResult(
                operation="register", path=path, pin=request.pin
            )
        if request.operation == "active":
            if request.now is None:  # unreachable past the request validator
                raise ValueError("active needs the instant to judge expiry against")
            return ModelDevLaneCodeRefPinResult(
                operation="active",
                path=pin_path(request.state_dir),
                pin=self.active(request.state_dir, request.now),
            )
        return ModelDevLaneCodeRefPinResult(
            operation="release", path=self.release(request.state_dir)
        )

    def register(self, state_dir: Path, pin: ModelDevLaneCodeRefPin) -> Path:
        return write_pin(state_dir, pin)

    def active(self, state_dir: Path, now: datetime) -> ModelDevLaneCodeRefPin | None:
        return read_active_pin(state_dir, now)

    def release(self, state_dir: Path) -> Path:
        return release_pin(state_dir)


def followed_branch(
    tracking_branch: str,
    env: Mapping[str, str],
    boundary: str,
    *,
    clock: Callable[[], datetime] | None = None,
) -> str:
    """The branch ``self_update`` follows: the live pin's, else the tracking branch.

    ``env`` is the agent's process environment, injected; the state directory is
    ``DEPLOY_AGENT_STATE_DIR``, which the unit declares. A unit that declares none has
    no pin.
    """
    state_dir = env.get(ENV_STATE_DIR, "").strip()
    if not state_dir:
        return tracking_branch
    pin = read_active_pin(Path(state_dir), (clock or (lambda: datetime.now(UTC)))())
    if pin is None:
        return tracking_branch
    logger.info(
        "self_update[boundary=%s]: following pinned branch %s instead of %s "
        "(pr=%s ticket=%s registered_head=%s expires=%s)",
        boundary,
        pin.branch,
        tracking_branch,
        pin.pr,
        pin.ticket,
        pin.head[:12],
        pin.expires_at.isoformat(),
    )
    return pin.branch
