# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21604 AC3 -- the dispatch port carries no 240 s default, and a class whose
ceiling plus margin does not fit the consumer dispatch deadline is refused when
the port loads.

The ceiling and margin come from the task-class authority (the contract, or the
site overlay that raises it); nothing here or in the port holds a number of its
own. A class the consumer's dispatch deadline would cancel mid-run is refused at
load, naming the class and both figures.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import cast

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.delegation.model_dispatch_budget import ModelDispatchBudget
from omnibase_infra.protocols.protocol_pattern_b_broker_transport import (
    ProtocolPatternBBrokerTransport,
)
from omnibase_infra.runtime import service_delegation_dispatch_port as port_module
from omnibase_infra.runtime.protocols import protocol_delegation_dispatch_port
from omnibase_infra.runtime.protocols.protocol_delegation_dispatch_port import (
    ProtocolDelegationDispatchPort,
)
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)
from tests.helpers.cli_registry_stand_in import (
    StandInExecutionBudget,
    install_stand_in_registry,
    wiring_authority,
)

pytestmark = pytest.mark.unit

_DEADLINE_SECONDS = 600.0


def _bus(deadline: float | None = _DEADLINE_SECONDS) -> ProtocolPatternBBrokerTransport:
    """A bus whose consume loop applies ``deadline`` (none: it exposes no deadline)."""
    config = SimpleNamespace(effective_dispatch_deadline_seconds=deadline)
    bus = SimpleNamespace() if deadline is None else SimpleNamespace(config=config)
    return cast("ProtocolPatternBBrokerTransport", bus)


def _authority(*, ceiling: int, margin: int = 60, cls: str = "document"):
    base = wiring_authority(terminal_delivery_margin_seconds=margin)
    budgets = dict(base.budgets)
    budgets[cls] = StandInExecutionBudget(
        task_class_timeout_ceiling_seconds=ceiling,
        terminal_delivery_margin_seconds=margin,
    )
    return base.model_copy(update={"budgets": budgets})


@pytest.mark.parametrize(
    "dispatch",
    [RuntimeDelegationDispatchPort.dispatch, ProtocolDelegationDispatchPort.dispatch],
    ids=["runtime port", "protocol"],
)
def test_the_budget_arguments_default_to_unset_not_to_a_number(dispatch) -> None:
    parameters = inspect.signature(dispatch).parameters
    for name in ("execution_timeout_seconds", "terminal_delivery_margin_seconds"):
        assert parameters[name].default is None, name


def test_no_default_budget_constants_remain() -> None:
    for module in (protocol_delegation_dispatch_port, port_module):
        assert not hasattr(module, "DEFAULT_EXECUTION_TIMEOUT_SECONDS")
        assert not hasattr(module, "DEFAULT_TERMINAL_DELIVERY_MARGIN_SECONDS")


def test_a_class_past_the_dispatch_deadline_is_refused_when_the_port_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Overlay raised `document` to 900 s: 900 + 60 > 600 s deadline.
    install_stand_in_registry(monkeypatch, _authority(ceiling=900))
    with pytest.raises(ProtocolConfigurationError) as raised:
        RuntimeDelegationDispatchPort(_bus())
    message = str(raised.value)
    assert "document" in message
    assert "960" in message
    assert "600" in message


def test_classes_inside_the_dispatch_deadline_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: the same load passes when the figures fit."""
    install_stand_in_registry(monkeypatch, _authority(ceiling=300))
    RuntimeDelegationDispatchPort(_bus())


def test_the_boundary_is_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    install_stand_in_registry(monkeypatch, _authority(ceiling=540))
    RuntimeDelegationDispatchPort(_bus())


def test_no_deadline_given_means_no_check(monkeypatch: pytest.MonkeyPatch) -> None:
    install_stand_in_registry(monkeypatch, _authority(ceiling=900))
    RuntimeDelegationDispatchPort(_bus(None))


def test_an_unset_budget_resolves_from_the_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_stand_in_registry(monkeypatch, _authority(ceiling=600, margin=45))
    assert ModelDispatchBudget.resolve("document", None, None) == (600, 45)


def test_a_caller_figure_is_kept_and_only_the_unset_one_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_stand_in_registry(monkeypatch, _authority(ceiling=600, margin=45))
    assert ModelDispatchBudget.resolve("document", 90, None) == (90, 45)
    assert ModelDispatchBudget.resolve("document", None, 7) == (600, 7)
    with pytest.raises(ValueError):
        ModelDispatchBudget.resolve("document", 0, None)
