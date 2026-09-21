# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test-only fixture handler that round-trips ``correlation_id`` (OMN-17295).

This module is NOT production code. It is the delegate-shaped sibling of
``handler_proof_noop``: its request model declares ``correlation_id``, so
``RuntimeLocal`` propagates the caller's id onto the terminal event exactly as
the real ``ModelDelegateSkillRequest`` does (that model is frozen, so
``RuntimeLocal``'s event-driven correlation overwrite is refused and the CLI's
minted id survives onto the wire — verified live).

``handler_proof_noop``'s request model declares no ``correlation_id`` at all,
so a run against it produces a terminal stamped with a runtime-minted id that
no caller can attribute. That is a legitimate refusal under the OMN-17295
correlation join, which makes it the wrong stand-in for tests whose subject is
the *shape* of the delegate receipt. Production handlers MUST NOT import from
or depend on this module.
"""

from __future__ import annotations

from hashlib import sha256

from pydantic import BaseModel

from omnibase_core.models.delegation.wire import (
    ModelDelegationBudgetEvidence,
    ModelDelegationContractEvidence,
)
from omnibase_infra.cli.model_delegate_attempt import ModelDelegateAttempt
from omnibase_infra.cli.model_delegate_terminal import ModelDelegateTerminal

_TASK_CLASS_TIMEOUT_CEILING_SECONDS = 240
_TERMINAL_DELIVERY_MARGIN_SECONDS = 60
_FIXTURE_RESPONSE_CONTRACT = b"correlated-noop fixture response contract: plain_text"
_FIXTURE_CONTRACT_SHA256 = sha256(_FIXTURE_RESPONSE_CONTRACT).hexdigest()


class ModelCorrelatedNoopRequest(BaseModel):
    """Test-only input model carrying the caller's correlation id."""

    correlation_id: str = ""
    prompt: str = ""
    task_type: str = ""
    requested_timeout_seconds: int | None = None


class ModelDelegateSkillFixtureTerminal(ModelDelegateTerminal):
    """Fixture-only terminal whose concrete name declares the delegate wire role."""


class HandlerCorrelatedNoop:
    """Emit the typed completed terminal the strict delegate receipt requires."""

    def handle(
        self, request: ModelCorrelatedNoopRequest
    ) -> ModelDelegateSkillFixtureTerminal:
        execution_timeout_seconds = (
            _TASK_CLASS_TIMEOUT_CEILING_SECONDS
            if request.requested_timeout_seconds is None
            else request.requested_timeout_seconds
        )
        return ModelDelegateSkillFixtureTerminal(
            attempts=(
                ModelDelegateAttempt(
                    tier="fixture",
                    backend_id="correlated-noop",
                    model_id="fixture",
                    quality_gate_passed=True,
                    acceptance_decision="accept",
                ),
            ),
            response=request.prompt,
            model_name="correlated-noop-fixture",
            provider="fixture://correlated-noop",
            status="completed",
            budget_evidence=ModelDelegationBudgetEvidence(
                requested_timeout_seconds=request.requested_timeout_seconds,
                task_class_timeout_ceiling_seconds=_TASK_CLASS_TIMEOUT_CEILING_SECONDS,
                execution_timeout_seconds=execution_timeout_seconds,
                terminal_delivery_margin_seconds=_TERMINAL_DELIVERY_MARGIN_SECONDS,
            ),
            response_contract_evidence=ModelDelegationContractEvidence(
                conveyed=True,
                validated=True,
                output_shape="plain_text",
                contract_sha256=_FIXTURE_CONTRACT_SHA256,
                channel="fixture",
            ),
        )
