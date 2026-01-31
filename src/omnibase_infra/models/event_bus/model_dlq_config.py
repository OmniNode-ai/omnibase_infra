# SPDX-FileCopyrightText: 2025 OmniNode Team <info@omninode.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""Dead Letter Queue configuration model for event bus message handling.

This module provides the configuration model for DLQ behavior in event bus
consumers. The DLQ routes messages that fail processing to a dead letter
topic for later analysis, retry, or manual intervention.

Error Classification:
    The DLQ configuration distinguishes between two error categories:

    Content Errors (non-retryable):
        Schema validation failures, malformed payloads, missing required fields,
        type conversion errors. These errors will NOT fix themselves with retry.
        Default behavior: Send to DLQ and commit offset (dlq_and_commit).

    Infrastructure Errors (potentially retryable):
        Database timeouts, network failures, service unavailability.
        These errors MAY fix themselves after retry budget exhaustion.
        Default behavior: Fail fast (fail_fast) to avoid hiding infrastructure
        fires in the DLQ.

Topic Naming Convention:
    When topic is empty string, the consumer builds a DLQ topic name
    following ONEX conventions: {env}.dlq.{original_topic}.v{schema_major}

    Examples:
        - dev.dlq.orders.created.v1
        - prod.dlq.payments.processed.v1
        - staging.dlq.users.registered.v1

See Also:
    - MixinKafkaDlq: DLQ publishing implementation
    - ModelDlqEvent: Individual DLQ event model for callbacks
    - docs/architecture/DLQ_MESSAGE_FORMAT.md: DLQ message structure
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelDlqConfig(BaseModel):
    """Dead Letter Queue (DLQ) configuration.

    Controls how failed messages are routed to DLQ topics. Supports
    differentiated handling of content errors (non-retryable) versus
    infrastructure errors (potentially retryable).

    Attributes:
        enabled: Whether DLQ publishing is enabled for failed messages.
            When False, failed messages are dropped (logged only).
            Default: True.
        topic: DLQ topic name. Empty string triggers convention-based
            topic naming: {env}.dlq.{original_topic}.v{schema_major}.
            Non-empty value overrides the convention with explicit topic.
            Default: "" (use convention).
        on_content_error: Action when a content/schema error occurs.
            Content errors (schema validation, malformed payload) are
            non-retryable - they will never succeed with retry.
            - "dlq_and_commit": Publish to DLQ and commit offset (default)
            - "fail_fast": Raise immediately, do not commit
            Default: "dlq_and_commit".
        on_infra_exhausted: Action when retry budget exhausted for
            infrastructure errors. Infrastructure errors (DB timeout,
            network failure) may fix themselves, but infra owns plumbing.
            - "dlq_and_commit": Publish to DLQ and commit offset
            - "fail_fast": Raise immediately, do not commit (default)
            Default: "fail_fast".

    Example:
        ```python
        from omnibase_infra.models.event_bus import ModelDlqConfig

        # Production configuration (explicit topic, fail-fast for infra)
        config = ModelDlqConfig(
            enabled=True,
            topic="",  # Use convention-based naming
            on_content_error="dlq_and_commit",
            on_infra_exhausted="fail_fast",
        )

        # Development configuration (catch everything in DLQ)
        dev_config = ModelDlqConfig(
            enabled=True,
            topic="dev.dlq.catch-all.v1",
            on_content_error="dlq_and_commit",
            on_infra_exhausted="dlq_and_commit",
        )

        # Disabled DLQ (for testing or specific use cases)
        disabled = ModelDlqConfig(enabled=False)
        ```

    Configuration Guidelines:
        - Enable DLQ for all production consumers to capture failures
        - Use "fail_fast" for on_infra_exhausted to surface infrastructure
          issues immediately rather than hiding them in DLQ
        - Use "dlq_and_commit" for on_content_error since content errors
          will never self-heal with retry
        - Set explicit topic only when you need multiple consumers to
          share a DLQ or when convention doesn't fit

    Design Rationale:
        Default on_content_error = "dlq_and_commit":
            Content errors (bad schema, malformed JSON) will never fix
            themselves. Retrying is pointless. Send to DLQ for human
            review and continue processing other messages.

        Default on_infra_exhausted = "fail_fast":
            Infrastructure owns the plumbing. If the database is down,
            that's an infrastructure fire that should be surfaced
            immediately - not hidden in a DLQ. The operations team needs
            to know about infrastructure failures, not discover them
            later in a DLQ audit.

    See Also:
        MixinKafkaDlq: Implementation of DLQ publishing behavior.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "enabled": True,
                    "topic": "",
                    "on_content_error": "dlq_and_commit",
                    "on_infra_exhausted": "fail_fast",
                },
                {
                    "enabled": True,
                    "topic": "prod.dlq.orders.v1",
                    "on_content_error": "dlq_and_commit",
                    "on_infra_exhausted": "dlq_and_commit",
                },
            ]
        },
    )

    enabled: bool = Field(
        default=True,
        description="Enable DLQ publishing for failed messages",
    )

    topic: str = Field(
        default="",
        description=(
            "DLQ topic name. Empty string uses convention-based naming: "
            "{env}.dlq.{original_topic}.v{schema_major}"
        ),
    )

    on_content_error: Literal["dlq_and_commit", "fail_fast"] = Field(
        default="dlq_and_commit",
        description=(
            "Action on content/schema errors (non-retryable). "
            "'dlq_and_commit' publishes to DLQ and commits offset. "
            "'fail_fast' raises immediately without committing."
        ),
    )

    on_infra_exhausted: Literal["dlq_and_commit", "fail_fast"] = Field(
        default="fail_fast",
        description=(
            "Action when retry budget exhausted for infrastructure errors. "
            "'fail_fast' surfaces infrastructure issues immediately. "
            "'dlq_and_commit' publishes to DLQ and commits offset."
        ),
    )


__all__ = ["ModelDlqConfig"]
