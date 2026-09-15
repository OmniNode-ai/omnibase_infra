# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canonical field binding tests for action-authorization nonce claims."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.action_authorization_claim import (
    ModelActionAuthorizationClaimRequest,
)


def _request(**overrides: object) -> ModelActionAuthorizationClaimRequest:
    values: dict[str, object] = {
        "authorization_id": "action-auth-12345678-1234-1234-1234-123456789abc",
        "ticket_id": "OMN-17462",
        "contract_path": "contracts/OMN-17462.yaml",
        "contract_commit_sha": "a" * 40,
        "contract_sha256": "sha256:" + "b" * 64,
        "action_id": "postgres-push-lane-bootstrap",
        "source_sha": "c" * 40,
        "artifact_sha256": "sha256:" + "d" * 64,
        "target_database": "rsd_push_lanes",
        "target_schema": "push_lanes",
        "target_service": "rsd_push_lane_broker",
        "target_principal": "rsd_push_lane_broker",
        "execute_enabled": False,
        "issuer": "operator-governance",
        "nonce": "e" * 64,
        "issued_at": datetime(2030, 1, 1, 10, tzinfo=UTC),
        "expires_at": datetime(2030, 1, 1, 11, tzinfo=UTC),
        "one_time_use": True,
        "reason": "bounded execute-disabled bootstrap verification",
    }
    values.update(overrides)
    return ModelActionAuthorizationClaimRequest.model_validate(values)


@pytest.mark.unit
def test_request_binds_exactly_the_nineteen_registry_fields() -> None:
    request = _request()

    assert set(request.canonical_registry_fields()) == {
        "authorization_id",
        "ticket_id",
        "contract_path",
        "contract_commit_sha",
        "contract_sha256",
        "action_id",
        "source_sha",
        "artifact_sha256",
        "target_database",
        "target_schema",
        "target_service",
        "target_principal",
        "execute_enabled",
        "issuer",
        "nonce",
        "issued_at",
        "expires_at",
        "one_time_use",
        "reason",
    }
    expected = json.dumps(
        request.canonical_registry_fields(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert (
        request.request_digest == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    )
    assert (
        request.nonce_digest
        == hashlib.sha256(request.nonce.encode("ascii")).hexdigest()
    )
    assert request.nonce not in request.sql_arguments()


@pytest.mark.unit
def test_any_canonical_field_change_changes_the_request_digest() -> None:
    request = _request()
    assert (
        _request(reason="different bounded reason").request_digest
        != request.request_digest
    )
    assert (
        _request(target_schema="other_schema").request_digest != request.request_digest
    )
    assert _request(nonce="f" * 64).request_digest != request.request_digest


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execute_enabled", True),
        ("one_time_use", False),
        ("contract_path", "contracts/OMN-99999.yaml"),
        ("expires_at", datetime(2030, 1, 1, 10, tzinfo=UTC)),
        ("target_principal", "*"),
    ],
)
def test_request_rejects_noncanonical_or_nondefault_deny_values(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        _request(**{field: value})
