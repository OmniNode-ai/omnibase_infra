# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Stand-in registry entries for the ``onex`` CLI tests (OMN-19407).

The CLI reads every vocabulary from the installed contract registry through
``omnibase_infra.cli.contract_registry._entry_points``. omnibase_infra's test
environment has no omnimarket (the venv-purity gate), so these helpers replace
that one seam with stand-in entries:

* ``onex.contracts:task_class_authority`` -> :class:`StandInTaskClassAuthority`,
  an object answering the same questions the real task-class authority does,
  over whatever classes the test declares;
* optionally ``onex.nodes:<delegate node>`` -> the stand-in node package next
  to this file, whose ``contract.yaml`` names its own input model.

Nothing here copies a production list. A test builds the authority it needs
and asserts the CLI offers exactly that, so the assertion is about the read
path rather than about any particular vocabulary.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.cli import contract_registry

__all__ = [
    "StandInExecutionBudget",
    "StandInTaskClassAuthority",
    "install_stand_in_registry",
    "use_stand_in_registry",
    "wiring_authority",
]

_STAND_IN_NODE_MODULE = "tests.helpers.cli_registry_stand_in.node_delegate_stand_in"
_AUTHORITY_LOADER = "tests.helpers.cli_registry_stand_in:current_authority"

_current: StandInTaskClassAuthority | None = None


class StandInExecutionBudget(BaseModel):
    """A declared handler budget."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_class_timeout_ceiling_seconds: int = Field(ge=1)
    terminal_delivery_margin_seconds: int = Field(ge=1)


class StandInSelectionFallback(BaseModel):
    """The class an unclaimed prompt resolves to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_class: str


class StandInResolution(BaseModel):
    """The class, how it was decided, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_type: str
    resolution: str
    reason: str


class StandInTaskClassAuthority(BaseModel):
    """Answers the questions ``onex delegate`` asks of the task-class authority.

    ``phrases`` maps a public class to words that claim a prompt for it (a
    whole-word presence test, first class by name wins); the real selection
    rules are the contract owner's and are tested there.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    public: frozenset[str]
    internal: frozenset[str] = frozenset()
    unroutable: dict[str, str] = Field(default_factory=dict)
    phrases: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    fallback: str | None = None
    budgets: dict[str, StandInExecutionBudget] = Field(default_factory=dict)

    @property
    def public_task_classes(self) -> frozenset[str]:
        return self.public

    @property
    def internal_task_classes(self) -> frozenset[str]:
        return self.internal | frozenset(self.unroutable)

    @property
    def unroutable_task_classes(self) -> dict[str, str]:
        return dict(self.unroutable)

    @property
    def selection_fallback(self) -> StandInSelectionFallback | None:
        if self.fallback is None:
            return None
        return StandInSelectionFallback(task_class=self.fallback)

    def resolve_task_type(
        self, prompt: str, *, explicit: str | None
    ) -> StandInResolution:
        if explicit is not None:
            if explicit in self.unroutable:
                raise ValueError(
                    f"task class {explicit!r} is declared but not routable: "
                    f"{self.unroutable[explicit]}"
                )
            if explicit in self.public or explicit in self.internal:
                return StandInResolution(
                    task_type=explicit,
                    resolution="explicit",
                    reason="explicitly selected with --task-type",
                )
            raise ValueError(f"unknown task type {explicit!r}")
        words = set(prompt.lower().split())
        for name in sorted(self.public):
            claimed = [
                phrase for phrase in self.phrases.get(name, ()) if phrase in words
            ]
            if claimed:
                return StandInResolution(
                    task_type=name,
                    resolution="contract",
                    reason=f"stand-in phrase {claimed[0]!r} claimed the prompt",
                )
        if self.fallback is None:
            raise ValueError("no phrase claimed the prompt and no fallback is declared")
        return StandInResolution(
            task_type=self.fallback,
            resolution="fallback",
            reason="no stand-in phrase claimed the prompt",
        )

    def execution_budget(self, task_class: str) -> StandInExecutionBudget:
        try:
            return self.budgets[task_class]
        except KeyError as exc:
            raise ValueError(f"no declared execution budget for {task_class}") from exc


def current_authority() -> StandInTaskClassAuthority:
    """The ``onex.contracts`` loader the stand-in entry point names."""
    if _current is None:
        raise FileNotFoundError("no stand-in task-class authority is installed")
    return _current


def wiring_authority(
    *, terminal_delivery_margin_seconds: int = 60
) -> StandInTaskClassAuthority:
    """The authority CLI-wiring tests run against.

    Its class names are the ones those tests pass as ``--task-type``; nothing
    compares them to a production contract.
    """
    names = frozenset(
        {
            "code_generation",
            "complex_reasoning",
            "document",
            "refactor",
            "research",
            "summarization",
        }
    )
    budget = StandInExecutionBudget(
        task_class_timeout_ceiling_seconds=240,
        terminal_delivery_margin_seconds=terminal_delivery_margin_seconds,
    )
    return StandInTaskClassAuthority(
        public=names,
        fallback="document",
        budgets=dict.fromkeys(names, budget),
    )


def install_stand_in_registry(
    monkeypatch: pytest.MonkeyPatch,
    authority: StandInTaskClassAuthority | None,
    *,
    delegate_node_name: str | None = None,
) -> None:
    """Replace the registry seam with stand-in entries for this test.

    ``authority=None`` advertises the task-class entry with a loader that
    fails, which is how a test proves the CLI refuses by name. With
    ``delegate_node_name`` the stand-in node package is advertised in
    ``onex.nodes`` under that name; every other installed entry is kept.
    """
    monkeypatch.setattr(
        "tests.helpers.cli_registry_stand_in._current", authority, raising=True
    )
    real = contract_registry._entry_points
    stand_in: dict[str, tuple[EntryPoint, ...]] = {
        contract_registry.CONTRACT_RESOURCE_GROUP: (
            EntryPoint(
                name="task_class_authority",
                value=_AUTHORITY_LOADER,
                group=contract_registry.CONTRACT_RESOURCE_GROUP,
            ),
        ),
    }
    if delegate_node_name is not None:
        stand_in[contract_registry.NODE_GROUP] = (
            EntryPoint(
                name=delegate_node_name,
                value=_STAND_IN_NODE_MODULE,
                group=contract_registry.NODE_GROUP,
            ),
        )

    def _with_stand_ins(group: str) -> tuple[EntryPoint, ...]:
        added = stand_in.get(group, ())
        replaced = {entry.name for entry in added}
        return tuple(e for e in real(group) if e.name not in replaced) + added

    monkeypatch.setattr(contract_registry, "_entry_points", _with_stand_ins)


def use_stand_in_registry(
    monkeypatch: pytest.MonkeyPatch,
    authority: StandInTaskClassAuthority | None = None,
) -> None:
    """Give ``onex delegate`` a stand-in authority and the stand-in request model.

    The delegate request model is resolved through the stand-in node's
    contract, the same read ``onex delegate`` performs, but without
    advertising a delegate node in ``onex.nodes`` -- tests that stub the
    delegate contract's topics keep doing so.
    """
    from omnibase_infra.cli import cli_delegate
    from tests.helpers.cli_registry_stand_in.node_delegate_stand_in.model_stand_in_delegate_request import (
        ModelStandInDelegateRequest,
    )

    install_stand_in_registry(
        monkeypatch, wiring_authority() if authority is None else authority
    )
    monkeypatch.setattr(
        cli_delegate, "_delegate_request_model", lambda: ModelStandInDelegateRequest
    )
