# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for the delegation identity check contract and doctor check."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import omnibase_infra.doctor.checks.check_delegation_identity as check_module
from omnibase_core.enums.enum_health_status_value import EnumHealthStatusValue
from omnibase_infra.cli.cli_delegate import DelegateTenantRefusedError
from omnibase_infra.doctor.checks.check_delegation_identity import (
    CheckDelegationIdentity,
)

pytestmark = pytest.mark.integration

UPSTREAM_TEXT: str = "cannot parse /srv/tenant/identity.yaml: line 3 column 9"
UPSTREAM_FRAGMENTS: tuple[str, ...] = ("/srv/tenant", "identity.yaml", "line 3")
FIX_HINT: str = "Fix: onex local init"


def _load_contract() -> dict[str, Any]:
    """Load the node contract YAML file relative to the repository root."""
    contract_path: Path = Path(__file__).resolve().parents[3] / (
        "src/omnibase_infra/nodes/node_check_delegation_identity_effect/contract.yaml"
    )
    with contract_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _import_model(module: str, name: str) -> type:
    """Import a class from a dotted module path."""
    module_obj: Any = importlib.import_module(module)
    return getattr(module_obj, name)


def _import_handler(module: str, name: str) -> Any:
    """Import a handler class from a dotted module path."""
    module_obj: Any = importlib.import_module(module)
    return getattr(module_obj, name)


def test_contract_input_model_feeds_the_contract_handler(tmp_path: Path) -> None:
    """Verify the contract's input model and default handler behave as specified."""
    contract: dict[str, Any] = _load_contract()

    input_model_spec: dict[str, str] = contract["input_model"]
    model_cls: type = _import_model(
        input_model_spec["module"], input_model_spec["name"]
    )
    assert model_cls.__name__ == "ModelDelegationIdentityRequest", (
        f"Expected input model class name 'ModelDelegationIdentityRequest', "
        f"got '{model_cls.__name__}'"
    )

    request: Any = model_cls(onex_home=tmp_path, identity_unreadable=True)

    with pytest.raises(ValidationError) as exc_info:
        model_cls(onex_home=tmp_path, identity_unreadable=True, unknown_extra_field="x")
    assert "unknown_extra_field" in str(exc_info.value), (
        "ValidationError message should mention the unknown extra field"
    )

    handler_spec: str = contract["handler_routing"]["default_handler"]
    handler_module: str = handler_spec.rsplit(":", 1)[0]
    handler_name: str = handler_spec.rsplit(":", 1)[1]
    handler_cls: Any = _import_handler(handler_module, handler_name)
    handler: Any = handler_cls()

    def _failing_gateway_reader(onex_home: Path) -> tuple[Any, Any]:
        raise AssertionError(
            "gateway_reader must not be called when identity_unreadable is True"
        )

    refusal_cause: BaseException | None = (
        RuntimeError(UPSTREAM_TEXT) if request.identity_unreadable else None
    )
    diagnosis, judged = handler.evaluate_identity(
        request.onex_home,
        request.local_identity,
        refusal_cause,
        _failing_gateway_reader,
    )

    assert diagnosis.fault is None, "fault must be None for an unreadable identity"
    assert judged is False, "judged flag must be False for an unreadable identity"
    assert FIX_HINT in diagnosis.detail, f"detail must contain '{FIX_HINT}'"
    for fragment in UPSTREAM_FRAGMENTS:
        assert fragment not in diagnosis.detail, (
            f"detail must not contain upstream fragment '{fragment}'"
        )


def test_unreadable_identity_through_the_real_doctor_check_never_echoes_upstream_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the doctor check suppresses upstream error text and reports the fix hint."""

    def _raising_read_install_identity() -> str:
        raise DelegateTenantRefusedError(UPSTREAM_TEXT) from RuntimeError(UPSTREAM_TEXT)

    monkeypatch.setattr(
        check_module, "read_install_identity", _raising_read_install_identity
    )

    check: CheckDelegationIdentity = CheckDelegationIdentity(onex_home=tmp_path)
    diagnosis: Any = check.diagnose()
    result: Any = check.run()

    for text in (diagnosis.detail, result.message):
        for fragment in UPSTREAM_FRAGMENTS:
            assert fragment not in text, (
                f"Output text must not contain upstream fragment '{fragment}': {text!r}"
            )
        assert FIX_HINT in text, f"Output text must contain '{FIX_HINT}': {text!r}"

    assert result.status is not EnumHealthStatusValue.HEALTHY, (
        "status must not be HEALTHY when identity is unreadable"
    )


def test_a_readable_identity_is_healthy_and_does_not_print_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a readable identity yields a healthy status without leaking the UUID."""
    uuid_value: str = "11111111-2222-3333-4444-555555555555"

    def _returning_read_install_identity() -> str:
        return uuid_value

    monkeypatch.setattr(
        check_module, "read_install_identity", _returning_read_install_identity
    )

    check: CheckDelegationIdentity = CheckDelegationIdentity(onex_home=tmp_path)
    result: Any = check.run()

    assert result.status is EnumHealthStatusValue.HEALTHY, (
        f"status must be HEALTHY for a readable identity, got {result.status!r}"
    )
    assert uuid_value not in result.message, (
        f"run().message must not contain the identity UUID: {result.message!r}"
    )
