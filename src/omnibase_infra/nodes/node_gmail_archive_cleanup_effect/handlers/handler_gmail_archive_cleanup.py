# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for Gmail archive cleanup — core logic for node_gmail_archive_cleanup_effect.

Implements the ``gmail.purge_archive`` operation declared in contract.yaml.

Cleanup Strategy
----------------
For each label in ``archive_labels``, the handler builds a Gmail search query:

    label:<label_name> before:<YYYY/MM/DD>

where the cutoff date is computed from ``retention_days`` relative to UTC now.
The search returns message IDs only — no per-message metadata is fetched
(no N+1 requests).

For each returned message ID, ``HandlerGmailApi.delete_message`` is called.
Failures on individual deletes are collected in ``errors`` and the loop
continues (soft failure). A ``search_messages`` failure causes the label to be
skipped entirely and sets ``hard_failed=True`` on the result.

Handler Purity
--------------
The handler does NOT publish events directly. Instead it returns
``ModelGmailCleanupResult.pending_events`` — a list of event payloads for
the node shell / runtime to publish. This follows the ONEX contract that
handlers must not access the event bus.

Event Contract
--------------
A single summary event is appended to ``pending_events`` when
``purged_count > 0`` OR any errors exist:

    {
        "event_type": "onex.evt.omnibase_infra.gmail-archive-purged.v1",
        "purged_count": <int>,
        "label_counts": {<label_name>: <count>, ...},
        "error_count": <int>,
        "partition_key": "gmail-archive-cleanup",
    }

Related Tickets:
    - OMN-2731: Add node_gmail_archive_cleanup_effect
    - OMN-2728: Gmail Integration epic (omnibase_infra)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import cast

from omnibase_core.types import JsonType
from omnibase_infra.handlers.handler_gmail_api import HandlerGmailApi
from omnibase_infra.nodes.node_gmail_archive_cleanup_effect.models.model_gmail_cleanup_config import (
    ModelGmailCleanupConfig,
)
from omnibase_infra.nodes.node_gmail_archive_cleanup_effect.models.model_gmail_cleanup_result import (
    ModelGmailCleanupResult,
)

logger = logging.getLogger(__name__)

__all__ = ["HandlerGmailArchiveCleanup"]

# Event type emitted per cleanup run
_PURGED_EVENT_TYPE = "onex.evt.omnibase_infra.gmail-archive-purged.v1"

# Partition key: all cleanup events share a single partition
_PARTITION_KEY = "gmail-archive-cleanup"


class HandlerGmailArchiveCleanup:
    """Handler for the ``gmail.purge_archive`` operation.

    Searches each configured archive label for messages older than
    ``retention_days`` and permanently deletes them via HandlerGmailApi.

    Handler Purity:
        This handler does NOT publish events directly. All event payloads
        are returned in ``ModelGmailCleanupResult.pending_events`` for the
        runtime to publish. Handlers must not access the event bus.

    Failure Semantics:
        - ``search_messages`` raises → label skipped, ``hard_failed=True``.
        - Individual ``delete_message`` failures → appended to ``errors``,
          loop continues (soft failure).
        - A single summary event is emitted when ``purged_count > 0`` or
          any errors exist.

    Args:
        gmail_api: Shared HandlerGmailApi instance providing the OAuth2 +
            REST transport layer. If not provided, a new instance is created
            using environment variable credentials.
    """

    def __init__(
        self,
        gmail_api: HandlerGmailApi | None = None,
    ) -> None:
        self._gmail = gmail_api if gmail_api is not None else HandlerGmailApi()

    async def handle(
        self,
        config: ModelGmailCleanupConfig,
    ) -> ModelGmailCleanupResult:
        """Execute one archive cleanup run for all configured labels.

        For each label in ``config.archive_labels``:
          1. Build a date-bounded Gmail search query.
          2. Retrieve message IDs via ``search_messages`` (no N+1 fetch).
          3. Delete each message; collect per-message failures in ``errors``.
          4. On ``search_messages`` failure: set ``hard_failed=True`` and
             skip the label.

        Emits a single summary event to ``pending_events`` when
        ``purged_count > 0`` or any errors exist.

        Args:
            config: Cleanup configuration (archive_labels, retention_days).

        Returns:
            ``ModelGmailCleanupResult`` with purge counts, per-label
            breakdown, errors, and pending event payloads.
        """
        cutoff_str = (
            datetime.now(UTC) - timedelta(days=config.retention_days)
        ).strftime("%Y/%m/%d")

        purged_count = 0
        label_counts: dict[str, int] = {}
        hard_failed = False
        errors: list[str] = []

        for label_name in config.archive_labels:
            query = f"label:{label_name} before:{cutoff_str}"

            # Search: failure → hard_failed, skip label
            try:
                message_stubs = await self._gmail.search_messages(query=query)
            except Exception as exc:
                hard_failed = True
                msg = (
                    f"search_messages failed for label '{label_name}': "
                    f"{type(exc).__name__}"
                )
                logger.warning("%s", msg)
                errors.append(msg)
                continue

            # search_messages returns [] on internal failure (soft path)
            # treat as 0 results — no hard_failed in this case per design
            label_deleted = 0
            for stub in message_stubs:
                message_id_raw = stub.get("id", "")
                message_id = str(message_id_raw) if message_id_raw is not None else ""
                if not message_id:
                    continue

                success = await self._gmail.delete_message(message_id)
                if success:
                    label_deleted += 1
                else:
                    msg = (
                        f"delete_message failed for message '{message_id}' "
                        f"in label '{label_name}'"
                    )
                    logger.warning("%s", msg)
                    errors.append(msg)

            label_counts[label_name] = label_deleted
            purged_count += label_deleted

        # Build pending_events: one summary event when anything happened
        pending_events: list[JsonType] = []
        if purged_count > 0 or errors:
            # Cast label_counts dict[str, int] to dict[str, JsonType] for event payload
            label_counts_json = cast("dict[str, JsonType]", label_counts)
            summary_event: JsonType = {
                "event_type": _PURGED_EVENT_TYPE,
                "purged_count": purged_count,
                "label_counts": label_counts_json,
                "error_count": len(errors),
                "partition_key": _PARTITION_KEY,
            }
            pending_events.append(summary_event)

        return ModelGmailCleanupResult(
            purged_count=purged_count,
            label_counts=label_counts,
            hard_failed=hard_failed,
            errors=errors,
            events_published=0,  # Runtime publishes from pending_events
            pending_events=pending_events,
        )
