# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's way in to the queued request priority handler (OMN-26718).

The agent runs from its own checkout beside a published ``omnibase_infra`` that does not carry
this handler, so it loads the handler and its models by path (as it does the dev boot guard).
``request`` turns the agent's queued commands into the handler's request.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_MODULES = (
    "models/model_queued_request_entry.py",
    "models/model_queued_request_priority_request.py",
    "models/model_queued_request_priority_decision.py",
    "handlers/queued_request_priority/handler_queued_request_priority.py",
)


def _load() -> ModuleType:
    for relative in _MODULES:
        name = "omnibase_infra." + relative.removesuffix(".py").replace("/", ".")
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                name, _PACKAGE_ROOT / relative
            )
            assert spec is not None and spec.loader is not None
            sys.modules[name] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


def handler_class() -> type[Any]:
    """The handler class; the caller builds and calls it."""
    handler_cls: type[Any] = _load().HandlerQueuedRequestPriority
    return handler_cls


def request(queued: Sequence[Any], *, supersede_queue: bool) -> Any:
    """The handler's request for ``queued`` commands, head first."""
    module = _load()
    entry = module.ModelQueuedRequestEntry
    return module.ModelQueuedRequestPriorityRequest(
        entries=tuple(
            entry(
                correlation_id=q.command.correlation_id,
                runtime_lane=q.command.runtime_lane.value,
                requested_by=q.command.requested_by,
                git_ref=q.command.git_ref,
                scope=q.command.scope.value,
                build_source=q.command.build_source.value,
                services=tuple(q.command.services),
                rollback_declared=q.command.rollback is not None,
                pinned_image=bool(q.command.image_ref or q.command.image_digest),
            )
            for q in queued
        ),
        supersede_queue=supersede_queue,
    )
