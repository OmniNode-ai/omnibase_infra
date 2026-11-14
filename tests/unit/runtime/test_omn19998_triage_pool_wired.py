# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998 -- a required dependency the runtime never supplies degrades the
handler into a silent suppressor.

``node_runtime_error_triage_effect``'s ``contract.yaml`` declares
``db_pool`` as ``required: true`` and ``HandlerRuntimeErrorTriage`` takes it
as a constructor parameter. The runtime's
``_build_runtime_handler_dependencies`` supplied the pool to
``HandlerPostgresRuntimeManifestInsert`` and ``HandlerBaselinesBatchCompute``
and never to the triage handler, so the auto-wired handler was built with
``db_pool=None`` and every one of the 1,291 runtime errors it saw in a
20-minute window logged ``db_pool not configured, suppressing event``. No
exception, no red check -- only a warning line that says exactly that.

The second test below is the mechanical form of the rule, discovered from
the handler's ``__init__`` signature and the contract's
``required: true`` dependencies: every dependency the node declares as
required and the handler takes as a constructor parameter must appear in the
map the resolver reads, or it resolves to ``None`` and the handler no-ops
silently.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import yaml

import omnibase_infra.nodes.node_runtime_error_triage_effect as _triage_pkg
from omnibase_infra.nodes.node_runtime_error_triage_effect.handlers.handler_runtime_error_triage import (
    HandlerRuntimeErrorTriage,
)
from omnibase_infra.runtime.service_kernel import _build_runtime_handler_dependencies

pytestmark = pytest.mark.unit

_HANDLER_NAME = "HandlerRuntimeErrorTriage"


def _contract_dependencies() -> dict[str, bool]:
    """Dependency name -> required, discovered from the node's contract.yaml."""
    contract_path = Path(str(_triage_pkg.__file__)).parent / "contract.yaml"
    contract: dict[str, Any] = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    return {
        dependency["name"]: bool(dependency.get("required", False))
        for dependency in contract["dependencies"]
    }


def _constructor_parameters() -> set[str]:
    """Every constructor parameter of the triage handler, defaulted or not.

    ``db_pool`` defaults to ``None``, which is exactly why a map that omits it
    fails nothing: the handler wires and suppresses.
    """
    signature = inspect.signature(HandlerRuntimeErrorTriage.__init__)
    return {name for name in signature.parameters if name != "self"}


def test_runtime_supplies_the_declared_db_pool() -> None:
    """RED before the fix: the map has no ``HandlerRuntimeErrorTriage`` entry
    at all, so the auto-wired handler is built with ``db_pool=None``."""
    pool = object()
    dependencies = _build_runtime_handler_dependencies(pool)

    assert dependencies is not None
    assert _HANDLER_NAME in dependencies, sorted(dependencies)
    assert dependencies[_HANDLER_NAME]["db_pool"] is pool, dependencies[_HANDLER_NAME]


def test_every_required_constructor_dependency_of_the_triage_handler_is_wired() -> None:
    """RED before the fix: ``db_pool`` is declared required and is a
    constructor parameter, but the runtime never supplies it.

    Discovered from the handler's ``__init__`` signature and the contract,
    so it covers the class rather than the one dependency that was missed:
    a future required dependency the kernel forgets is red here instead of
    silent in the lane.
    """
    required = _contract_dependencies()
    assert "db_pool" in required, (
        "guard on the guard: this test is vacuous if the contract stops "
        "declaring db_pool"
    )

    params = _constructor_parameters()
    targets = {
        name for name, is_required in required.items() if is_required and name in params
    }
    assert targets, (
        "guard on the guard: no required contract dependency is a "
        "constructor parameter, so there is nothing to check"
    )

    dependencies = _build_runtime_handler_dependencies(object())
    assert dependencies is not None
    wired = set(dependencies.get(_HANDLER_NAME, {}))

    missing = sorted(targets - wired)
    assert not missing, (
        f"{_HANDLER_NAME} declares {sorted(targets)} required in contract.yaml "
        f"but the runtime supplies {sorted(wired)}; {missing} would resolve to "
        "None and suppress every runtime error silently"
    )


def test_no_pool_leaves_the_triage_entry_absent() -> None:
    """A runtime with no pool must not plant a ``HandlerRuntimeErrorTriage``
    entry at all; absent is the honest shape of 'nothing to give'."""
    dependencies = _build_runtime_handler_dependencies(None)

    assert _HANDLER_NAME not in (dependencies or {}), dependencies
