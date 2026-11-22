# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host composition root of the deploy job reading effect (OMN-24617).

The deploy agent loads this by file path (``load_dev_boot_guard``). It loads the handler
and the two models the handler imports from the same checkout, reads the node's
``settings`` from the contract of the agent's own checkout, binds the handler to the process environment and
hands it the agent's settled job record. The handler and the contract hold the names, the
bounds and the append route; nothing here repeats them.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, JsonValue, ValidationError

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_deploy_job_reading_effect.runtime_post_failure_converge import (
        PostFailureConverge,
    )
    from omnibase_infra.protocols.protocol_command import ProtocolCommand
    from omnibase_infra.protocols.protocol_post_failure_host import (
        ProtocolPostFailureHost,
    )

logger = logging.getLogger(__name__)

NODE = "src/omnibase_infra/nodes/node_deploy_job_reading_effect"
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
#: The modules the handler imports, then the handler, in import order.
_CHECKOUT_MODULES = (
    "models/model_deploy_job_reading_request.py",
    "models/model_deploy_job_reading_result.py",
    "handlers/handler_deploy_job_reading.py",
)


def _load(relative: str) -> ModuleType:
    """One module of this checkout, loaded under its dotted name by file path, once."""
    name = "omnibase_infra." + relative.removesuffix(".py").replace("/", ".")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _PACKAGE_ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the deploy job reading module: {relative}")
    module = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


request_module, result_module, handler_module = (
    _load(relative) for relative in _CHECKOUT_MODULES
)

_POST_FAILURE_MODULES = (
    "enums/enum_post_failure_verdict.py",
    "enums/enum_job_status.py",
    "models/model_job_command.py",
    "models/model_job_lineage.py",
    "models/model_job_record.py",
    "models/model_failed_dev_job.py",
    "models/model_post_failure_inputs.py",
    "models/model_post_failure_decision.py",
    "protocols/protocol_command.py",
    "protocols/protocol_job_store.py",
    "protocols/protocol_lane_flags.py",
    "protocols/protocol_post_failure_host.py",
    "nodes/node_deploy_job_reading_effect/runtime_post_failure_converge.py",
)


def post_failure_converge(
    host: ProtocolPostFailureHost, converge_command: Callable[[], ProtocolCommand]
) -> PostFailureConverge:
    """The agent's post-failure converge rule (OMN-26429), loaded from this checkout once.

    It reads the same job records this node turns into readings, and shares nothing else.
    """
    *_, rule = (_load(relative) for relative in _POST_FAILURE_MODULES)
    converge: PostFailureConverge = rule.PostFailureConverge(host, converge_command)
    return converge


def contract_path(execution: ModuleType) -> Path:
    """The node's contract in the checkout the deploy agent's own module was loaded from."""
    module_file = execution.__file__
    if not module_file:
        raise RuntimeError(
            "the deploy job reading needs the deploy agent source location"
        )
    return Path(module_file).resolve().parents[3] / NODE / "contract.yaml"


def load_settings(execution: ModuleType) -> dict[str, JsonValue]:
    """The node contract's ``settings`` block: names, bounds and the append route variables."""
    return _contract_settings(contract_path(execution))


def _contract_settings(path: Path) -> dict[str, JsonValue]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = contract["settings"]
    if not isinstance(settings, dict):
        raise RuntimeError("the deploy job reading contract declares no settings")
    loaded: dict[str, JsonValue] = settings
    return loaded


def publish(job: BaseModel, execution: ModuleType) -> BaseModel | None:
    """Write the reading row of the deploy agent's settled ``job`` record; never raises.

    ``job`` is read through its JSON form so this module holds no import of the agent;
    ``execution`` is an agent module, which locates the checkout the contract is read from.
    Returns the handler's result, or ``None`` when the record could not be read.
    """
    try:
        record = json.loads(job.model_dump_json())
        request = request_module.ModelDeployJobReadingRequest(
            correlation_id=record["correlation_id"],
            runtime_lane=record["command"]["runtime_lane"],
            scope=record["command"]["scope"],
            job_outcome=record["status"],
            phase_results=record["phase_results"],
            not_started=record.get("not_started") or "",
            first_error=(record.get("errors") or [""])[0],
            completed_at=record.get("completed_at"),
            published_at=datetime.now(UTC),
        )
        handler = handler_module.HandlerDeployJobReading(
            settings=load_settings(execution), environ=os.environ
        )
        result: BaseModel = handler.handle(request)
        return result
    except (KeyError, TypeError, ValueError, ValidationError, RuntimeError, OSError):
        logger.exception("job reading skipped: the job record could not be read")
        return None


_pin_last_check: list[float] = []
_pin_settings: list[dict[str, JsonValue]] = []
_PIN_HANDLER = "omnibase_infra.handlers.handler_dev_lane_code_ref_pin"
#: Until the contract has been read once, the throttle holds the check this long between tries.
_PIN_RETRY_SECONDS = 300.0


def check_pin(*, force: bool = False) -> BaseModel | None:
    """Alarm when the agent's live code-ref pin is inside the window before its expiry; never raises.

    Called from the agent's idle tick, so it runs on a schedule with no job. The pin record is
    read from the agent's own state directory (the pin handler's ``DEPLOY_AGENT_STATE_DIR``),
    through the pin handler the agent loaded before this module. The contract is read and
    parsed once per process; the check is throttled to its
    ``pin_alarm_check_interval_seconds`` (``force`` skips the throttle), and the throttle is
    stamped before any read, so a failing read logs once per interval, not once per tick.
    Returns the handler's result, or ``None`` when throttled, when no pin is live or on an
    unreadable record.
    """
    try:
        tick = time.monotonic()
        interval = (
            _pin_settings[0]["pin_alarm_check_interval_seconds"]
            if _pin_settings
            else _PIN_RETRY_SECONDS
        )
        if (
            not force
            and _pin_last_check
            and isinstance(interval, int | float)
            and tick - _pin_last_check[0] < interval
        ):
            return None
        _pin_last_check[:] = [tick]
        if not _pin_settings:
            _pin_settings[:] = [
                _contract_settings(_PACKAGE_ROOT.parents[1] / NODE / "contract.yaml")
            ]
        settings = _pin_settings[0]
        pin_handler = sys.modules[_PIN_HANDLER]
        handler = handler_module.HandlerDeployJobReading(
            settings=settings, environ=os.environ
        )
        state_dir = handler.pin_state_dir()
        now = datetime.now(UTC)
        pin = pin_handler.read_active_pin(state_dir, now) if state_dir else None
        if pin is None:
            return None
        result: BaseModel = handler.handle_pin(
            branch=pin.branch,
            pr=pin.pr,
            ticket=pin.ticket,
            registered_by=pin.registered_by,
            expires_at=pin.expires_at,
            now=now,
        )
        return result
    except (KeyError, TypeError, ValueError, ValidationError, RuntimeError, OSError):
        logger.exception(
            "pin alarm check skipped: the pin or contract could not be read"
        )
        return None
