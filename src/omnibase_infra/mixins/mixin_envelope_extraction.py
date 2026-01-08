# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Envelope extraction mixin for infrastructure handlers.

This module provides a reusable mixin for extracting correlation_id and envelope_id
from request envelopes. These IDs are essential for distributed tracing and
causality tracking across infrastructure components.

Envelope Structure:
    Request envelopes in ONEX infrastructure follow a standard structure::

        {
            "envelope_id": UUID,         # Unique ID for this specific envelope
            "correlation_id": UUID,      # Groups related operations across services
            "operation": str,            # Operation identifier (e.g., "http.get")
            "payload": dict,             # Operation-specific payload data
            "metadata": dict,            # Optional metadata (timestamps, sources)
        }

    The envelope_id and correlation_id fields can be either UUID objects or
    string representations. This mixin handles both formats transparently.

Features:
    - Extracts correlation_id for request grouping and distributed tracing
    - Extracts envelope_id for request/response causality tracking
    - Handles both UUID objects and string representations
    - Auto-generates UUIDs when extraction fails (graceful degradation)
    - Type-safe extraction with proper UUID validation

Usage:
    ```python
    from omnibase_infra.mixins import MixinEnvelopeExtraction

    class MyHandler(MixinEnvelopeExtraction):
        async def handle(self, envelope: dict[str, object]):
            correlation_id = self._extract_correlation_id(envelope)
            envelope_id = self._extract_envelope_id(envelope)
            # ... use IDs for tracing and causality tracking
    ```

Correlation ID vs Envelope ID:
    - correlation_id: Groups related operations across services in a distributed system.
      All operations triggered by a single user request share the same correlation_id.
    - envelope_id: Links a specific response to its originating request envelope.
      Enables fine-grained request/response pairing even across async boundaries.
"""

from uuid import UUID, uuid4


class MixinEnvelopeExtraction:
    """Mixin providing envelope extraction utilities for handlers.

    Provides standardized methods to extract correlation_id and envelope_id
    from request envelopes, with automatic fallback to generated UUIDs.

    This mixin should be inherited by infrastructure handlers that need to
    extract tracing IDs from incoming request envelopes.
    """

    def _extract_correlation_id(self, envelope: dict[str, object]) -> UUID:
        """Extract or generate correlation ID from envelope.

        Correlation IDs enable distributed tracing by grouping all operations
        triggered by a single user request. This method extracts the correlation_id
        from the request envelope if present and valid, otherwise generates a new one.

        Args:
            envelope: Request envelope dict that may contain correlation_id
                as either a UUID object or a string representation.

        Returns:
            UUID: The extracted correlation_id if valid, otherwise a newly generated UUID.
            Auto-generation ensures all requests have valid tracing IDs even when
            the request omits correlation_id.
        """
        raw = envelope.get("correlation_id")
        if isinstance(raw, UUID):
            return raw
        if isinstance(raw, str):
            try:
                return UUID(raw)
            except ValueError:
                pass
        return uuid4()

    def _extract_envelope_id(self, envelope: dict[str, object]) -> UUID:
        """Extract or generate envelope ID for causality tracking.

        Envelope IDs enable end-to-end causality tracking in distributed systems.
        Unlike correlation_id (which groups related operations across services),
        envelope_id links a specific response to its originating request envelope.

        Causality Chain Example:
            Request envelope (envelope_id=A) -> Handler processes -> Response (input_envelope_id=A)

        This allows observability systems to:
            - Trace request/response pairs even across async boundaries
            - Identify which response corresponds to which request
            - Build complete request flow graphs in distributed tracing

        Args:
            envelope: Request envelope dict that may contain envelope_id
                as either a UUID object or a string representation.

        Returns:
            UUID: The extracted envelope_id if valid, otherwise a newly generated UUID.
            Auto-generation ensures all responses have valid causality tracking IDs
            even when the request omits envelope_id.
        """
        raw = envelope.get("envelope_id")
        if isinstance(raw, UUID):
            return raw
        if isinstance(raw, str):
            try:
                return UUID(raw)
            except ValueError:
                pass
        return uuid4()
