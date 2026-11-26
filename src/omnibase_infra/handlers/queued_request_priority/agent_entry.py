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
from typing import Protocol

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_MODULES = (
    "models/model_queued_request_entry.py",
    "models/model_queued_request_priority_request.py",
    "models/model_queued_request_priority_decision.py",
    "handlers/queued_request_priority/handler_queued_request_priority.py",
)


class _Valued(Protocol):
    @property
    def value(self) -> str: ...


class _Command(Protocol):
    """The fields of a queued deploy command the priority request reads."""

    correlation_id: str
    runtime_lane: _Valued
    requested_by: str
    git_ref: str
    scope: _Valued
    build_source: _Valued
    services: Sequence[str]
    rollback: object | None
    image_ref: str | None
    image_digest: str | None


class _Queued(Protocol):
    @property
    def command(self) -> _Command: ...


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


def handler_class() -> type:
    """The handler class; the caller builds and calls it."""
    handler_cls: type = _load().HandlerQueuedRequestPriority
    return handler_cls


def request(queued: Sequence[_Queued], *, supersede_queue: bool) -> object:
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
