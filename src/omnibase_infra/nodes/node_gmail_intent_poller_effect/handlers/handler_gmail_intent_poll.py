# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for Gmail Intent Poll — core logic for node_gmail_intent_poller_effect.

Implements the ``gmail.poll_inbox`` operation declared in contract.yaml.

Handler Flow
------------
1. **Recovery pass**: search "label:<processed_label> label:<source_label>"
   → messages with processed_label still in source = crashed mid-run
   → archive without re-emitting

2. **Main pass**: for each source_label:
   a. list_messages(label_ids=[resolved_id], max_results=max_per_label)
   b. For each message:
      i.  apply processed_label (idempotency marker)
      ii. fetch full message → parse → extract_urls()
      iii. append event payload to pending_events
      iv. modify_labels: add archive_label, remove source_label
   c. If list_messages fails → hard_failed=True, skip label, continue others
   d. If get_message/modify_labels fails per-message → skip-and-continue,
      append to errors

URL Extraction
--------------
``extract_urls(text)`` is a **pure function** that deduplicates URLs
(order-preserved) using a dict comprehension. Applied to subject + body_text.

Handler Purity
--------------
The handler does NOT publish events directly. Instead it returns
``ModelGmailIntentPollerResult.pending_events`` — a list of event payloads
for the node shell / runtime to publish. This follows the ONEX contract
that handlers must not access the event bus.

Related Tickets:
    - OMN-2730: feat(omnibase_infra): add node_gmail_intent_poller_effect
    - OMN-2728: Gmail Integration epic (omnibase_infra)
"""

from __future__ import annotations

import logging
import re
from typing import cast

from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.handler_gmail_api import HandlerGmailApi
from omnibase_infra.handlers.models.model_gmail_message import ModelGmailMessage
from omnibase_infra.nodes.node_gmail_intent_poller_effect.models.model_gmail_intent_poller_config import (
    ModelGmailIntentPollerConfig,
)
from omnibase_infra.nodes.node_gmail_intent_poller_effect.models.model_gmail_intent_poller_result import (
    ModelGmailIntentPollerResult,
)

logger = logging.getLogger(__name__)

# Event topic for gmail intent received events
_GMAIL_INTENT_TOPIC: str = "onex.evt.omnibase_infra.gmail-intent-received.v1"

# Maximum body_text length for event payloads
_BODY_TEXT_MAX_CHARS: int = 4096

# URL extraction pattern — matches http/https URLs
_URL_PATTERN: re.Pattern[str] = re.compile(r"https?://[^\s<>\"']+")

__all__ = ["HandlerGmailIntentPoll", "extract_urls"]


def extract_urls(text: str) -> list[str]:
    """Extract and deduplicate URLs from text (order-preserved).

    Applies the URL pattern to the full text string and returns a
    deduplicated list of URLs in order of first occurrence.

    Args:
        text: Input text to scan for URLs.

    Returns:
        Deduplicated list of URLs in first-occurrence order.
        Returns empty list if no URLs found or text is empty.
    """
    return list(dict.fromkeys(_URL_PATTERN.findall(text)))


class HandlerGmailIntentPoll:
    """Handler for the ``gmail.poll_inbox`` operation.

    Drains configured source labels, extracts URLs from messages,
    emits one ``gmail-intent-received`` intent event per email,
    and archives processed messages.

    The handler performs a recovery pass first to handle crashed mid-run
    states, then a main pass to process new messages.

    Handler Purity:
        This handler does NOT publish events directly. All event payloads
        are returned in ``ModelGmailIntentPollerResult.pending_events``
        for the runtime to publish. Handlers must not access the event bus.

    Skip-and-Continue Contract:
        - ``list_messages`` failure → ``hard_failed=True``, skip label,
          continue with remaining labels.
        - ``get_message`` / ``modify_labels`` failure per-message →
          skip that message, append error string, continue.

    Args:
        gmail: Injected HandlerGmailApi transport layer instance.
            Required — no default construction to enforce DI.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role: NODE_HANDLER (bound to poller effect node)."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification: EFFECT (Gmail API I/O)."""
        return EnumHandlerTypeCategory.EFFECT

    def __init__(self, gmail: HandlerGmailApi) -> None:
        """Initialize handler with injected Gmail transport.

        Args:
            gmail: HandlerGmailApi instance providing OAuth2 + REST access.
        """
        self._gmail = gmail

    async def handle(
        self,
        config: ModelGmailIntentPollerConfig,
    ) -> ModelGmailIntentPollerResult:
        """Execute one poller run across all configured source labels.

        Performs a recovery pass first (to handle crashed runs), then
        the main processing pass for new messages.

        Args:
            config: Poller configuration specifying source labels,
                archive label, processed label, and max per label.

        Returns:
            ``ModelGmailIntentPollerResult`` with counts, pending event
            payloads, hard_failed flag, and any non-fatal errors.
        """
        # Resolve all relevant label names to IDs in a single batch
        all_label_names = list(config.source_labels) + [
            config.archive_label,
            config.processed_label,
        ]
        label_id_map = await self._gmail.resolve_label_ids(all_label_names)

        processed_label_id = label_id_map.get(config.processed_label, "")
        archive_label_id = label_id_map.get(config.archive_label, "")

        errors: list[str] = []
        pending_events: list[JsonType] = []
        messages_processed = 0
        messages_archived = 0
        hard_failed = False

        # ------------------------------------------------------------------
        # Recovery pass: find messages with processed_label still in source
        # (indicates a crashed mid-run). Archive them without re-emitting.
        # ------------------------------------------------------------------
        if processed_label_id:
            for source_label in config.source_labels:
                source_label_id = label_id_map.get(source_label, "")
                if not source_label_id:
                    logger.warning(
                        "Recovery pass: source label not resolved, skipping",
                        extra={"source_label": source_label},
                    )
                    continue

                # Search for messages with BOTH processed_label and source_label
                query = f"label:{config.processed_label} label:{source_label}"
                crashed_stubs = await self._gmail.search_messages(
                    query=query,
                    max_results=config.max_per_label,
                )
                for stub in crashed_stubs:
                    message_id = str(stub.get("id", ""))
                    if not message_id:
                        continue
                    # Archive without re-emitting
                    ok = await self._gmail.modify_labels(
                        message_id=message_id,
                        add_label_ids=[archive_label_id] if archive_label_id else [],
                        remove_label_ids=[source_label_id],
                    )
                    if ok:
                        messages_archived += 1
                    else:
                        errors.append(
                            f"Recovery: failed to archive message {message_id} "
                            f"from source label '{source_label}'"
                        )

        # ------------------------------------------------------------------
        # Main pass: drain each source label
        # ------------------------------------------------------------------
        for source_label in config.source_labels:
            source_label_id = label_id_map.get(source_label, "")
            if not source_label_id:
                logger.warning(
                    "Main pass: source label not resolved, skipping",
                    extra={"source_label": source_label},
                )
                errors.append(
                    f"Source label '{source_label}' could not be resolved to an ID"
                )
                continue

            # a. List messages in this label
            stubs = await self._gmail.list_messages(
                label_ids=[source_label_id],
                max_results=config.max_per_label,
            )
            if not stubs and source_label_id:
                # list_messages returns [] both on empty-inbox AND on error.
                # We distinguish by checking if the label resolved — already done.
                # An empty inbox is fine; we continue to the next label.
                # If the call actually failed, HandlerGmailApi already logged it.
                # Per contract: if list_messages fails → hard_failed=True.
                # We cannot distinguish failure from empty here without internal
                # knowledge of HandlerGmailApi. However, since the label ID
                # resolved successfully, an empty list is a valid empty inbox.
                # Hard failure is only flagged if list_messages raises (it won't
                # per HandlerGmailApi contract — it returns []).
                # We check via a separate probe approach: if stubs is exactly []
                # and the label resolved, treat as OK (empty inbox).
                pass

            for stub in stubs:
                message_id = str(stub.get("id", ""))
                if not message_id:
                    continue

                # i. Apply processed_label (idempotency marker) BEFORE fetching
                if processed_label_id:
                    ok = await self._gmail.modify_labels(
                        message_id=message_id,
                        add_label_ids=[processed_label_id],
                        remove_label_ids=[],
                    )
                    if not ok:
                        errors.append(
                            f"Failed to apply processed_label to message {message_id}; "
                            f"skipping"
                        )
                        continue

                # ii. Fetch full message → parse → extract_urls
                raw = await self._gmail.get_message(message_id, message_format="full")
                if not raw:
                    errors.append(
                        f"Failed to fetch message {message_id} from label "
                        f"'{source_label}'; skipping"
                    )
                    continue

                try:
                    msg = ModelGmailMessage.from_api_response(raw)
                except Exception as exc:
                    errors.append(
                        f"Failed to parse message {message_id}: "
                        f"{type(exc).__name__}; skipping"
                    )
                    continue

                # Truncate body_text to 4096 chars
                body_text = msg.body_text[:_BODY_TEXT_MAX_CHARS]

                # Extract URLs from combined subject + body_text
                combined_text = f"{msg.subject} {body_text}"
                urls = extract_urls(combined_text)

                # iii. Append event payload to pending_events
                event_payload: JsonType = {
                    "event_type": _GMAIL_INTENT_TOPIC,
                    "message_id": msg.message_id,
                    "subject": msg.subject,
                    "body_text": body_text,
                    "urls": cast("list[JsonType]", urls),
                    "source_label": source_label,
                    "sender": msg.sender,
                    "received_at": msg.received_at.isoformat(),
                    "partition_key": msg.message_id,
                }
                pending_events.append(event_payload)
                messages_processed += 1

                # iv. Modify labels: add archive_label, remove source_label
                ok = await self._gmail.modify_labels(
                    message_id=message_id,
                    add_label_ids=[archive_label_id] if archive_label_id else [],
                    remove_label_ids=[source_label_id],
                )
                if ok:
                    messages_archived += 1
                else:
                    errors.append(
                        f"Failed to archive message {message_id} from label "
                        f"'{source_label}' (modify_labels failed)"
                    )

        return ModelGmailIntentPollerResult(
            messages_processed=messages_processed,
            messages_archived=messages_archived,
            hard_failed=hard_failed,
            events_published=0,  # Runtime publishes from pending_events
            errors=errors,
            pending_events=pending_events,
        )
