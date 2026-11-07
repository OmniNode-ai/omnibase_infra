# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21536: each flipped handler takes exactly one typed request.

One test per flipped node. Each asserts only the signature, so it runs on the
pre-change tree and fails there on its own assertion, which is what the
canonical-shape flip proof requires of a parity test.
"""

import inspect
import typing

from pydantic import BaseModel

from omnibase_infra.handlers.handler_validation_adjudicate import (
    HandlerAdjudicate,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding import (
    HandlerOnboarding,
)
from omnibase_infra.nodes.node_pattern_lifecycle_effect.handlers.handler_lifecycle_update import (
    HandlerLifecycleUpdate,
)
from omnibase_infra.nodes.node_rrh_emit_effect.handlers.handler_repo_state_collect import (
    HandlerRepoStateCollect,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.handlers.handler_rsd_data_fetch import (
    HandlerRsdDataFetch,
)
from omnibase_infra.nodes.node_rsd_orchestrator.handlers.handler_rsd_initiate import (
    HandlerRsdInitiate,
)
from omnibase_infra.nodes.node_scope_file_read_effect.handlers.handler_scope_file_read import (
    HandlerScopeFileRead,
)
from omnibase_infra.nodes.node_scope_manifest_write_effect.handlers.handler_scope_manifest_write import (
    HandlerScopeManifestWrite,
)
from omnibase_infra.nodes.node_setup_orchestrator.handlers.handler_setup_orchestrator import (
    HandlerSetupOrchestrator,
)
from omnibase_infra.nodes.node_validation_executor.handlers.handler_run_checks import (
    HandlerRunChecks,
)
from omnibase_infra.nodes.node_validation_ledger_write_effect.handlers.handler_validation_ledger_append import (
    HandlerValidationLedgerAppend,
)
from omnibase_infra.nodes.node_validation_orchestrator.handlers.handler_build_plan import (
    HandlerBuildPlan,
)


def _assert_one_typed_request(handler_class: type) -> None:
    handle = inspect.getattr_static(handler_class, "handle")
    parameters = tuple(
        name for name in inspect.signature(handle).parameters if name != "self"
    )
    assert parameters == ("request",), parameters
    request_type = typing.get_type_hints(handle)["request"]
    assert issubclass(request_type, BaseModel), request_type


def test_onboarding_orchestrator_handle_takes_one_typed_request() -> None:
    _assert_one_typed_request(HandlerOnboarding)


def test_pattern_lifecycle_effect_handle_takes_one_typed_request() -> None:
    _assert_one_typed_request(HandlerLifecycleUpdate)


def test_rrh_emit_effect_handle_takes_one_typed_request() -> None:
    _assert_one_typed_request(HandlerRepoStateCollect)


def test_rsd_data_fetch_effect_handle_takes_one_typed_request() -> None:
    _assert_one_typed_request(HandlerRsdDataFetch)


def test_rsd_orchestrator_handle_takes_one_typed_request() -> None:
    _assert_one_typed_request(HandlerRsdInitiate)


def test_scope_file_read_effect_handle_takes_one_typed_request() -> None:
    _assert_one_typed_request(HandlerScopeFileRead)


def test_scope_manifest_write_effect_handle_takes_one_typed_request() -> None:
    _assert_one_typed_request(HandlerScopeManifestWrite)


def test_setup_orchestrator_handle_takes_one_typed_request() -> None:
    _assert_one_typed_request(HandlerSetupOrchestrator)


def test_validation_adjudicator_handle_takes_one_typed_request() -> None:
    _assert_one_typed_request(HandlerAdjudicate)


def test_validation_executor_handle_takes_one_typed_request() -> None:
    _assert_one_typed_request(HandlerRunChecks)


def test_validation_ledger_write_effect_handle_takes_one_typed_request() -> None:
    _assert_one_typed_request(HandlerValidationLedgerAppend)


def test_validation_orchestrator_handle_takes_one_typed_request() -> None:
    _assert_one_typed_request(HandlerBuildPlan)
