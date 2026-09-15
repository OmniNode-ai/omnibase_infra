# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canonical, one-time pre-execution action-authorization request."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256Digest = Annotated[
    str,
    Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]
RegistrySha256Digest = Annotated[
    str,
    Field(min_length=71, max_length=71, pattern=r"^sha256:[0-9a-f]{64}$"),
]
AuthorizationId = Annotated[
    str,
    Field(
        min_length=48,
        max_length=48,
        pattern=(
            r"^action-auth-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12}$"
        ),
    ),
]
TicketId = Annotated[str, Field(pattern=r"^OMN-[1-9][0-9]*$")]
CommitSha = Annotated[
    str, Field(min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$")
]
ActionId = Annotated[
    str, Field(min_length=3, max_length=64, pattern=r"^[a-z][a-z0-9-]{2,63}$")
]
Identity = Annotated[
    str,
    Field(min_length=1, max_length=63, pattern=r"^[a-z0-9][a-z0-9_-]{0,62}$"),
]
Issuer = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"),
]
Nonce = Annotated[
    str, Field(min_length=32, max_length=128, pattern=r"^[0-9a-f]{32,128}$")
]


class ModelActionAuthorizationClaimRequest(BaseModel):
    """All canonical registry fields, retained only in redacted durable form.

    The raw ``nonce`` is used only to derive the nonce and request digests. It
    is intentionally excluded from SQL arguments and durable records.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    authorization_id: AuthorizationId
    ticket_id: TicketId
    contract_path: str = Field(min_length=1, max_length=256)
    contract_commit_sha: CommitSha
    contract_sha256: RegistrySha256Digest
    action_id: ActionId
    source_sha: CommitSha
    artifact_sha256: RegistrySha256Digest
    target_database: Identity
    target_schema: Identity
    target_service: Identity
    target_principal: Identity
    execute_enabled: bool
    issuer: Issuer
    nonce: Nonce
    issued_at: datetime
    expires_at: datetime
    one_time_use: bool
    reason: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def _validate_canonical_constraints(self) -> ModelActionAuthorizationClaimRequest:
        if self.contract_path != f"contracts/{self.ticket_id}.yaml":
            msg = "contract_path must match the canonical ticket contract"
            raise ValueError(msg)
        if self.execute_enabled is not False:
            msg = "execute_enabled must be false for a pre-execution authorization"
            raise ValueError(msg)
        if self.one_time_use is not True:
            msg = "one_time_use must be true for a nonce claim"
            raise ValueError(msg)
        if self.issued_at.tzinfo is None or self.expires_at.tzinfo is None:
            msg = "issued_at and expires_at must be timezone-aware instants"
            raise ValueError(msg)
        if self.expires_at <= self.issued_at:
            msg = "expires_at must be strictly after issued_at"
            raise ValueError(msg)
        if not self.reason.strip():
            msg = "reason must not be whitespace only"
            raise ValueError(msg)
        return self

    @staticmethod
    def _canonical_instant(value: datetime) -> str:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    def canonical_registry_fields(self) -> dict[str, object]:
        """Return the exact registry field set in canonical JSON-ready form."""
        return {
            "action_id": self.action_id,
            "artifact_sha256": self.artifact_sha256,
            "authorization_id": self.authorization_id,
            "contract_commit_sha": self.contract_commit_sha,
            "contract_path": self.contract_path,
            "contract_sha256": self.contract_sha256,
            "execute_enabled": self.execute_enabled,
            "expires_at": self._canonical_instant(self.expires_at),
            "issued_at": self._canonical_instant(self.issued_at),
            "issuer": self.issuer,
            "nonce": self.nonce,
            "one_time_use": self.one_time_use,
            "reason": self.reason,
            "source_sha": self.source_sha,
            "target_database": self.target_database,
            "target_principal": self.target_principal,
            "target_schema": self.target_schema,
            "target_service": self.target_service,
            "ticket_id": self.ticket_id,
        }

    @property
    def nonce_digest(self) -> Sha256Digest:
        """Return the only nonce representation supplied to PostgreSQL."""
        return hashlib.sha256(self.nonce.encode("ascii")).hexdigest()

    @property
    def request_digest(self) -> Sha256Digest:
        """Hash the exact canonical serialization of all registry fields."""
        canonical = json.dumps(
            self.canonical_registry_fields(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def redacted_receipt_digest(self) -> Sha256Digest:
        """Return a deterministic receipt identifier with no raw registry data."""
        material = f"action-authorization-claim-receipt.v1:{self.request_digest}"
        return hashlib.sha256(material.encode("ascii")).hexdigest()

    def sql_arguments(self) -> tuple[object, ...]:
        """Return the ordered, redacted PostgreSQL function arguments."""
        return (
            self.authorization_id,
            self.ticket_id,
            self.contract_path,
            self.contract_commit_sha,
            self.contract_sha256,
            self.action_id,
            self.source_sha,
            self.artifact_sha256,
            self.target_database,
            self.target_schema,
            self.target_service,
            self.target_principal,
            self.execute_enabled,
            self.issuer,
            self.nonce_digest,
            self.issued_at,
            self.expires_at,
            self.one_time_use,
            self.reason,
            self.request_digest,
            self.redacted_receipt_digest,
        )


__all__ = [
    "ModelActionAuthorizationClaimRequest",
    "RegistrySha256Digest",
    "Sha256Digest",
]
