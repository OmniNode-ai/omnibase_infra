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
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import yaml
from pydantic import BaseModel, JsonValue, ValidationError

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
    contract = yaml.safe_load(contract_path(execution).read_text(encoding="utf-8"))
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
