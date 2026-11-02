# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Project validator run and triggered events into validation ledger intents (OMN-22002).

omnibase_core publishes one run event per validation report and one triggered
event per refusing validator. This handler turns each into exactly one
``validation_ledger.append`` intent. The existing ``HandlerValidationLedgerAppend``
(bound through the projection node's ``intent_routing_table``) is the only writer
of ``validation_event_ledger``, so the table is fed one way.

The row keeps the raw event, base64-encoded in ``envelope_bytes``; the validator
id, repository, commit and verdict of a triggered event are read from it. The row's
own columns carry the correlation id as ``run_id``, the repository as ``repo_id``
and the Kafka position that makes a replay of the same message a duplicate.

No duration is recorded: neither wire model carries one (``ModelValidatorRunEvent``
has a single ``emitted_at``, ``ModelValidatorTriggeredEvent`` has no timestamp).

The wire model is chosen by the topic the message arrived on, never by trial
parsing: a payload that does not validate against its own topic's model is kept,
with fallback ``run_id`` and ``repo_id``, so the ledger never drops what the broker
delivered.

``occurred_at`` has one source per event type, because the wire models differ: a
run event carries ``emitted_at`` (a naive value is read as UTC); a triggered event
carries no timestamp, so the message header time is used. A message without a
Kafka partition and offset is refused, because those two numbers, with the topic,
are the row's idempotency key and a default would merge distinct messages.

No topic is named here: the node contract declares the subscriptions and routes
each topic to this handler, and the topic arrives on the consumed message.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import ValidationError

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_core.models.validation.model_validator_run_event import (
    ModelValidatorRunEvent,
)
from omnibase_core.models.validation.model_validator_triggered_event import (
    ModelValidatorTriggeredEvent,
)
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.enums.generated.enum_platform_topic import EnumPlatformTopic
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.intent_target_uris import load_intent_target_uris
from omnibase_infra.models.validation_ledger import (
    ModelPayloadValidationLedgerAppend,
)

HANDLER_ID_VALIDATOR_EVENT_LEDGER_PROJECT: str = (
    "validator-event-ledger-project-handler"
)
_UNKNOWN_REPOSITORY = "unknown"


class HandlerValidatorEventLedgerProject:
    """Pure projection of a validator wire event into one ledger append intent."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, request: ModelEventMessage
    ) -> ModelHandlerOutput[ModelIntent]:
        message = self._coerce(request)
        partition, offset = self._position(message)
        run_id, repo_id, occurred_at = self._row_identity(message)
        payload = ModelPayloadValidationLedgerAppend(
            run_id=run_id,
            repo_id=repo_id,
            event_type=message.topic,
            event_version=self._version(message.topic),
            occurred_at=occurred_at,
            kafka_topic=message.topic,
            kafka_partition=partition,
            kafka_offset=offset,
            envelope_bytes=base64.b64encode(message.value).decode("ascii"),
            envelope_hash=hashlib.sha256(message.value).hexdigest(),
            correlation_id=message.headers.correlation_id,
        )
        targets = load_intent_target_uris()
        intent = ModelIntent(
            intent_type=payload.intent_type,
            target=targets.target(
                targets.validation_event_ledger,
                payload.kafka_topic,
                payload.kafka_partition,
                payload.kafka_offset,
            ),
            payload=payload,
        )
        return ModelHandlerOutput.for_compute(
            input_envelope_id=message.headers.message_id,
            correlation_id=message.headers.correlation_id,
            handler_id=HANDLER_ID_VALIDATOR_EVENT_LEDGER_PROJECT,
            result=intent,
        )

    @staticmethod
    def _row_identity(message: ModelEventMessage) -> tuple[UUID, str, datetime]:
        """Return (run_id, repo_id, occurred_at) for the event on ``message``."""
        header_time = message.headers.timestamp
        fallback = (
            uuid5(NAMESPACE_URL, hashlib.sha256(message.value).hexdigest()),
            _UNKNOWN_REPOSITORY,
            header_time,
        )
        try:
            if message.topic == EnumPlatformTopic.EVT_VALIDATOR_TRIGGERED_V1.value:
                triggered = ModelValidatorTriggeredEvent.model_validate_json(
                    message.value
                )
                return triggered.correlation_id, triggered.repository, header_time
            if message.topic != EnumPlatformTopic.EVT_VALIDATOR_RUN_V1.value:
                return fallback
            run = ModelValidatorRunEvent.model_validate_json(message.value)
        except ValidationError:
            return fallback
        try:
            run_id = UUID(run.report_ref)
        except ValueError:
            run_id = uuid5(NAMESPACE_URL, run.report_ref)
        try:
            emitted = datetime.fromisoformat(run.emitted_at)
        except ValueError:
            emitted = header_time
        if emitted.tzinfo is None:
            emitted = emitted.replace(tzinfo=UTC)
        return run_id, _UNKNOWN_REPOSITORY, emitted

    @staticmethod
    def _coerce(raw: object) -> ModelEventMessage:
        if isinstance(raw, ModelEventMessage):
            return raw
        payload = raw.get("payload", raw) if isinstance(raw, dict) else raw
        if isinstance(payload, ModelEventMessage):
            return payload
        return ModelEventMessage.model_validate(payload)

    @staticmethod
    def _position(message: ModelEventMessage) -> tuple[int, int]:
        """Return (partition, offset); refuse a message that has no broker position."""
        try:
            if message.partition is None or message.offset is None:
                raise ValueError("partition and offset are both required")
            return message.partition, int(message.offset)
        except ValueError as exc:
            raise ProtocolConfigurationError(
                "validator event has no usable Kafka partition and offset, which key its ledger row",
                context=ModelInfraErrorContext.with_correlation(
                    correlation_id=message.headers.correlation_id,
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="project_validator_event",
                ),
            ) from exc

    @staticmethod
    def _version(topic: str) -> str:
        tail = topic.rsplit(".", maxsplit=1)[-1]
        return tail if tail.startswith("v") and tail[1:].isdigit() else "unknown"


__all__ = ["HandlerValidatorEventLedgerProject"]
