# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for NodeDeltaBundleEffect payload models.

Tests model validation, serialization, and immutability for
ModelPayloadWriteBundle and ModelPayloadUpdateOutcome.

Related Tickets:
    - OMN-3142: NodeDeltaBundleEffect implementation
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_delta_bundle_effect.models.model_payload_update_outcome import (
    ModelPayloadUpdateOutcome,
)
from omnibase_infra.nodes.node_delta_bundle_effect.models.model_payload_write_bundle import (
    ModelPayloadWriteBundle,
)


@pytest.mark.unit
class TestModelPayloadWriteBundle:
    """Tests for ModelPayloadWriteBundle."""

    def test_minimal_valid_payload(self) -> None:
        """Minimal required fields produce a valid payload."""
        payload = ModelPayloadWriteBundle(
            correlation_id=uuid4(),
            bundle_id=uuid4(),
            pr_ref="owner/repo#1",
            head_sha="abc123",
            base_sha="def456",
            gate_decision="PASS",
        )
        assert payload.intent_type == "delta_bundle.write_bundle"
        assert payload.coding_model is None
        assert payload.subsystem is None
        assert payload.gate_violations == []
        assert payload.labels == []

    def test_full_payload(self) -> None:
        """All fields populated produce a valid payload."""
        payload = ModelPayloadWriteBundle(
            correlation_id=uuid4(),
            bundle_id=uuid4(),
            pr_ref="owner/repo#42",
            head_sha="abc123",
            base_sha="def456",
            coding_model="claude-opus-4-20250514",
            subsystem="omnibase_infra",
            gate_decision="QUARANTINE",
            gate_violations=[{"rule": "test_coverage", "threshold": 80}],
            labels=["stabilizes:owner/repo#10", "bug"],
        )
        assert payload.gate_decision == "QUARANTINE"
        assert len(payload.gate_violations) == 1
        assert len(payload.labels) == 2

    def test_frozen_model(self) -> None:
        """Model is frozen (immutable)."""
        payload = ModelPayloadWriteBundle(
            correlation_id=uuid4(),
            bundle_id=uuid4(),
            pr_ref="owner/repo#1",
            head_sha="abc",
            base_sha="def",
            gate_decision="PASS",
        )
        with pytest.raises(ValidationError):
            payload.pr_ref = "changed"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError):
            ModelPayloadWriteBundle(
                correlation_id=uuid4(),
                bundle_id=uuid4(),
                pr_ref="owner/repo#1",
                head_sha="abc",
                base_sha="def",
                gate_decision="PASS",
                unexpected_field="boom",  # type: ignore[call-arg]
            )

    def test_invalid_gate_decision(self) -> None:
        """Invalid gate_decision value is rejected."""
        with pytest.raises(ValidationError):
            ModelPayloadWriteBundle(
                correlation_id=uuid4(),
                bundle_id=uuid4(),
                pr_ref="owner/repo#1",
                head_sha="abc",
                base_sha="def",
                gate_decision="INVALID",  # type: ignore[arg-type]
            )


@pytest.mark.unit
class TestModelPayloadUpdateOutcome:
    """Tests for ModelPayloadUpdateOutcome."""

    def test_minimal_valid_payload(self) -> None:
        """Minimal required fields produce a valid payload."""
        payload = ModelPayloadUpdateOutcome(
            correlation_id=uuid4(),
            pr_ref="owner/repo#1",
            head_sha="abc123",
            outcome="merged",
        )
        assert payload.intent_type == "delta_bundle.update_outcome"
        assert payload.merged_at is None

    def test_with_merged_at(self) -> None:
        """Payload with merged_at timestamp."""
        now = datetime.now(tz=UTC)
        payload = ModelPayloadUpdateOutcome(
            correlation_id=uuid4(),
            pr_ref="owner/repo#1",
            head_sha="abc123",
            outcome="merged",
            merged_at=now,
        )
        assert payload.merged_at == now

    def test_invalid_outcome(self) -> None:
        """Invalid outcome value is rejected."""
        with pytest.raises(ValidationError):
            ModelPayloadUpdateOutcome(
                correlation_id=uuid4(),
                pr_ref="owner/repo#1",
                head_sha="abc123",
                outcome="abandoned",  # type: ignore[arg-type]
            )

    def test_frozen_model(self) -> None:
        """Model is frozen (immutable)."""
        payload = ModelPayloadUpdateOutcome(
            correlation_id=uuid4(),
            pr_ref="owner/repo#1",
            head_sha="abc",
            outcome="closed",
        )
        with pytest.raises(ValidationError):
            payload.outcome = "merged"  # type: ignore[misc]
