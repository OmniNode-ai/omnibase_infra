# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for schema_version field annotation backfill [OMN-9025].

Verifies that models with string-version-ok annotations instantiate correctly
and that schema_version fields accept plain string values as required by their
wire format contracts (Kafka envelopes, cross-service consumers, YAML manifests).
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestSchemaVersionAnnotationBackfill:
    """Verify annotated schema_version: str fields work correctly at runtime."""

    def test_model_event_headers_schema_version_is_str(self) -> None:
        from datetime import UTC, datetime

        from omnibase_infra.event_bus.models.model_event_headers import (
            ModelEventHeaders,
        )

        headers = ModelEventHeaders(
            source="test",
            event_type="test.event",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert isinstance(headers.schema_version, str)
        assert headers.schema_version == "1.0.0"

    def test_model_event_headers_accepts_custom_version(self) -> None:
        from datetime import UTC, datetime

        from omnibase_infra.event_bus.models.model_event_headers import (
            ModelEventHeaders,
        )

        headers = ModelEventHeaders(
            source="test",
            event_type="test.event",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            schema_version="2.1.0",
        )
        assert headers.schema_version == "2.1.0"

    def test_model_stub_comment_schema_version_is_str(self) -> None:
        from datetime import UTC, datetime

        from omnibase_infra.adapters.project_tracker.model_stub_comment import (
            ModelStubComment,
        )

        comment = ModelStubComment(
            id="c-1",
            body="stub body",
            author="agent",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert isinstance(comment.schema_version, str)
        assert comment.schema_version == "1.0"

    def test_model_event_registry_fingerprint_element_schema_version_is_str(
        self,
    ) -> None:
        from omnibase_infra.runtime.emit_daemon.model_event_registry_fingerprint_element import (
            ModelEventRegistryFingerprintElement,
        )

        element = ModelEventRegistryFingerprintElement(
            event_type="test.event.created",
            topic_template="onex.evt.test.event-created.v1",
            schema_version="1.0.0",
            partition_key_field="correlation_id",
            required_fields=("correlation_id", "payload"),
            element_sha256="a" * 64,
        )
        assert isinstance(element.schema_version, str)
        assert element.schema_version == "1.0.0"
