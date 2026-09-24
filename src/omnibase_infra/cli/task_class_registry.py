# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The task-class contract, read through the registry for ``onex delegate`` (OMN-19407).

WHAT THIS REPLACED. ``task_class_selection.py`` parsed omnimarket's
``task_class_contracts.v1.yaml`` with its own raw-YAML loader into mirror
models of the contract's schema, evaluated selection itself, owned the
fallback class and a default execution budget, and was pinned to the contract
by hand-copied fixtures and two halves of a digest. Every one of those was a
second copy of something the contract already owns.

WHAT IT IS NOW. The contract's owner advertises a loader in the
``onex.contracts`` entry-point group as ``task_class_authority``; this module
reads it through :func:`contract_resource` and asks the returned authority
every question the CLI has: its public and internal classes, the classes it
declares unroutable, how a prompt resolves (selection, fallback, explicit
admission and refusal) and each class's execution budget.
:class:`ProtocolTaskClassAuthority` states which members the CLI calls.
"""

from __future__ import annotations

from omnibase_infra.cli.contract_registry import (
    RegistryUnresolvedError,
    contract_resource,
)
from omnibase_infra.cli.protocol_task_class_authority import (
    ProtocolTaskClassAuthority,
)
from omnibase_infra.cli.protocol_task_type_resolution import (
    ProtocolTaskTypeResolution,
)

__all__ = [
    "TASK_CLASS_AUTHORITY_RESOURCE",
    "TaskClassContractError",
    "describe_task_classes",
    "load_task_class_authority",
    "resolve_task_class",
]

#: The ``onex.contracts`` entry the task-class contract is advertised under.
TASK_CLASS_AUTHORITY_RESOURCE = "task_class_authority"


class TaskClassContractError(Exception):
    """The task-class contract is unreadable, or it refused this class or prompt.

    Raised, never defaulted: a class nobody declared is the defect OMN-18305
    removed.
    """


def load_task_class_authority() -> ProtocolTaskClassAuthority:
    """Read the task-class contract from the registry, or refuse by name."""
    try:
        authority = contract_resource(TASK_CLASS_AUTHORITY_RESOURCE)
    except RegistryUnresolvedError as exc:
        raise TaskClassContractError(str(exc)) from exc
    if not isinstance(authority, ProtocolTaskClassAuthority):
        raise TaskClassContractError(
            f"onex.contracts:{TASK_CLASS_AUTHORITY_RESOURCE} returned "
            f"{type(authority).__module__}.{type(authority).__qualname__}, which "
            "does not answer the questions onex delegate asks (public and "
            "internal classes, unroutable classes, the selection fallback, "
            "resolve_task_type and execution_budget). The installed contract "
            "owner predates this CLI; upgrade it"
        )
    return authority


def resolve_task_class(
    authority: ProtocolTaskClassAuthority, prompt: str, *, explicit: str | None
) -> ProtocolTaskTypeResolution:
    """Ask the authority for this run's class, mapping its refusal to this CLI's error."""
    try:
        return authority.resolve_task_type(prompt, explicit=explicit)
    except ValueError as exc:
        raise TaskClassContractError(str(exc)) from exc


def describe_task_classes(authority: ProtocolTaskClassAuthority) -> str:
    """Render the ``--task-type`` help from the contract, at invocation time."""
    unroutable = sorted(authority.unroutable_task_classes)
    internal = sorted(authority.internal_task_classes - set(unroutable))
    fallback = authority.selection_fallback
    return (
        "Task class for routing, read from the task-class contract "
        "(onex.contracts:task_class_authority) when this command runs. "
        f"Public, selectable from a prompt: {', '.join(sorted(authority.public_task_classes))}. "
        f"Internal, by explicit name only: {', '.join(internal) or '(none)'}. "
        f"Declared but not routable, refused with the contract's reason: "
        f"{', '.join(unroutable) or '(none)'}. Omit to resolve it from the "
        "contract's selection predicates; an unclaimed prompt resolves to the "
        "contract's selection_fallback"
        + (f" ({fallback.task_class})" if fallback is not None else " (none declared)")
        + ". The chosen class and how it was chosen are printed on stderr and "
        "recorded in the run artifacts."
    )
