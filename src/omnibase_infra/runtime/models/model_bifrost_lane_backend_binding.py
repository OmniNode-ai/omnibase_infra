# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One Bifrost backend binding declared by a lane overlay.

OMN-17099 replaced the authorization table this model used to validate against.
Until then the model refused any ``backend_id`` outside a hardcoded map of lab
hosts, ports and served ids, and the overlay demanded set equality with it, so
the shipped product carried lab values and a lab host could not be registered
without a release. The model is now a SCHEMA: it checks that a binding is
well-formed and complete, and the lane overlay file that declares it is the
authority on what the lab serves. Two checks that used to live in the table
live elsewhere now:

* served id — the renderer still refuses a binding whose ``served_model_id``
  disagrees with the base contract's ``model_name`` for the same backend, and
  ``tests/unit/runtime/test_bifrost_served_model_probe_fixture.py`` pins every
  committed lab overlay row to a recorded ``/v1/models`` probe;
* liveness — ``serving`` is declared by the overlay row and pinned to the same
  probe record, so a dark rung is still one flag, never a deleted row.

A binding either OVERRIDES a backend the base contract declares (endpoint,
served id, output ceiling, timeout) or ADDS one the base does not. An added
backend must carry the declaration the base contract would otherwise supply —
``provider``, ``tier`` and ``credential`` — all three, or none. Which of the two
cases applies is only knowable against the base contract, so the renderer
enforces it; this model enforces that the declaration is never partial.
"""

from __future__ import annotations

from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.runtime.models.enum_bifrost_lane_credential_kind import (
    EnumBifrostLaneCredentialKind,
)
from omnibase_infra.runtime.models.model_bifrost_lane_backend_credential import (
    ModelBifrostLaneBackendCredential,
)
from omnibase_infra.runtime.models.model_bifrost_lane_backend_placement import (
    ModelBifrostLaneBackendPlacement,
)

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
_ALLOWED_SCHEMES = frozenset({"http", "https"})

#: The fields an ADDED backend declares and a base-declared backend inherits
#: from the base contract. All-or-none on one binding.
NEW_BACKEND_DECLARATION_FIELDS: tuple[str, ...] = ("provider", "tier", "credential")


class ModelBifrostLaneBackendBinding(BaseModel):
    """One delegation backend binding from a lane overlay."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
        populate_by_name=True,
    )

    backend_key: str = Field(
        alias="backend_id",
        serialization_alias="backend_id",
        min_length=1,
    )
    endpoint_url: str = Field(min_length=1)
    advertised_model: str = Field(
        alias="served_model_id",
        serialization_alias="served_model_id",
        min_length=1,
    )
    parameter_count: str = Field(min_length=1)
    context_window: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    timeout_ms: int = Field(gt=0)
    #: OMN-16999. ``False`` renders the backend with ``endpoint_url: null`` —
    #: the shape the routing reducer skips — so a declared-but-dark rung stays
    #: in the artifact and is restored by flipping this flag.
    serving: bool = True
    #: OMN-17099. Declared only by a backend the base contract does not carry.
    provider: str | None = Field(default=None, min_length=1)
    tier: str | None = Field(default=None, min_length=1)
    credential: ModelBifrostLaneBackendCredential | None = None
    capabilities: tuple[str, ...] = ()
    #: OMN-19215. Where an ADDED backend sits in the routing tier ladder; passed
    #: through to the rendered contract. None keeps it reachable by pin only.
    placement: ModelBifrostLaneBackendPlacement | None = None

    @property
    def declares_new_backend(self) -> bool:
        """Whether this binding carries the declaration of an added backend."""
        return self.provider is not None

    @model_validator(mode="after")
    def _validate_binding(self) -> Self:
        if self.max_tokens > self.context_window:
            raise ValueError("max_tokens must not exceed context_window")

        declared = [
            name
            for name in NEW_BACKEND_DECLARATION_FIELDS
            if getattr(self, name) is not None
        ]
        if declared and len(declared) != len(NEW_BACKEND_DECLARATION_FIELDS):
            missing = [
                name for name in NEW_BACKEND_DECLARATION_FIELDS if name not in declared
            ]
            raise ValueError(
                f"backend {self.backend_key!r} declares {declared} but not "
                f"{missing}: a backend a lane adds must declare all of "
                f"{list(NEW_BACKEND_DECLARATION_FIELDS)} (credential is a "
                "secret_ref or an explicit kind 'none'), and a backend the base "
                "contract declares must declare none of them"
            )
        if self.capabilities and not declared:
            raise ValueError(
                f"backend {self.backend_key!r} declares capabilities without "
                "being an added backend: the base contract owns a declared "
                "backend's capabilities"
            )
        if self.placement is not None:
            if not declared:
                raise ValueError(
                    f"backend {self.backend_key!r} declares a placement without "
                    "being an added backend: a base-declared backend is already "
                    "in the routing ladder (OMN-19215)"
                )
            if self.placement.max_context_tokens > self.context_window:
                raise ValueError(
                    f"backend {self.backend_key!r} placement offers "
                    f"{self.placement.max_context_tokens} context tokens but the "
                    f"backend declares a context_window of {self.context_window} "
                    "(OMN-19215)"
                )

        parsed = urlsplit(self.endpoint_url)
        try:
            parsed.port  # noqa: B018 - raises ValueError on a malformed port
        except ValueError as exc:
            raise ValueError(
                f"endpoint_url for {self.backend_key!r} has an invalid port"
            ) from exc
        if (
            parsed.scheme not in _ALLOWED_SCHEMES
            or not parsed.hostname
            or parsed.path != _CHAT_COMPLETIONS_PATH
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                f"endpoint_url for {self.backend_key!r} must be a complete "
                f"http(s) endpoint ending in {_CHAT_COMPLETIONS_PATH} with a "
                "host; userinfo, query, and fragment are forbidden, got "
                f"{self.endpoint_url!r}"
            )
        if (
            self.credential is not None
            and self.credential.kind is EnumBifrostLaneCredentialKind.SECRET_REF
            and parsed.scheme != "https"
        ):
            raise ValueError(
                f"endpoint_url for {self.backend_key!r} authenticates with "
                f"secret_ref {self.credential.secret_ref!r} and must use https: "
                "a credential is never sent over plaintext http"
            )
        return self


__all__ = [
    "NEW_BACKEND_DECLARATION_FIELDS",
    "ModelBifrostLaneBackendBinding",
]
