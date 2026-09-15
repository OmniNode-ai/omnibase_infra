# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18385 -- a dead-letter envelope must not carry a credential.

RED ON ``origin/dev``. ``MixinKafkaDlq._publish_raw_to_dlq`` copied the raw
inbound record body verbatim into ``original_message.value``, and
``ModelGatewayAttachRequest.access_token`` was a plain ``str``. Live read of
``onex.dlq.omnibase-infra.commands.v1`` on onex-dev (partition 0, 96 records,
values fingerprinted and never printed): offsets 137 and 138 were dead-lettered
``gateway-attach-request`` records carrying ``payload.access_token`` in
cleartext, 1318 characters each, two distinct customer bearer tokens. Every
attach on that cluster had been dead-lettering since 2026-09-11 (OMN-16504), so
every customer attach in the window put its bearer token on a durable topic for
the topic's full retention.

TWO LAYERS, TESTED SEPARATELY.

AC1 (typing) closes the model-shaped paths: the credential fields are
``SecretStr``, so anything that serialises the MODEL emits a mask. The tests
for it assert the sentinel is absent from ``model_dump``/``model_dump_json``
and that the handler can still read the real value.

AC3 (publisher redaction) closes the raw path, which never constructs the
model -- it copies the bytes that failed to become one. Typing alone provably
does not close it, and the AC3 tests here are written against the RAW bytes for
exactly that reason.

Every test uses a SENTINEL token, never a real one. The assertion is that the
sentinel does not appear anywhere in the published bytes while the correlation
id does -- the record must stay useful for forensics after it stops being
dangerous.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr

from omnibase_infra.event_bus.mixin_kafka_dlq import MixinKafkaDlq
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_request import (
    ModelGatewayAttachRequest,
)
from omnibase_infra.utils.util_dlq_credential_redaction import (
    DLQ_REDACTION_MARKER,
    is_credential_field_name,
    redact_credential_fields,
    redact_credential_fields_in_text,
)

pytestmark = pytest.mark.unit

# onex-topic-allow: quotes the live topic the leaked records were read from
_DLQ_TOPIC = "onex.dlq.omnibase-infra.commands.v1"
# onex-topic-allow: quotes the live topic whose records were dead-lettered
_ORIGINAL_TOPIC = "onex.cmd.omnibase-infra.gateway-attach-request.v1"

# A sentinel, never a credential. Distinctive enough that a substring search
# over the published bytes cannot match it by accident.
_SENTINEL_TOKEN = "OMN18385-SENTINEL-TOKEN-ffffffffffffffffffffffff"
# The positive control: a NON-credential field whose value must SURVIVE. A
# redactor that passes the sentinel test by blanking the whole body would fail
# this one.
_SENTINEL_SURVIVES = "OMN18385-SENTINEL-EDGE-INSTANCE-must-survive"


class _CapturingProducer:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_and_wait(
        self,
        topic: str,
        *,
        value: bytes,
        key: bytes | None = None,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> object:
        self.sent.append(
            {"topic": topic, "value": value, "key": key, "headers": headers}
        )
        return object()


class _DlqHost(MixinKafkaDlq):
    """Minimal host exposing exactly what the mixin declares it needs."""

    def __init__(self) -> None:
        self._config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            dead_letter_topic=_DLQ_TOPIC,
        )
        self._environment = "test"
        self._group = "test-group"
        self._producer = _CapturingProducer()  # type: ignore[assignment]
        self._producer_lock = asyncio.Lock()
        self._timeout_seconds = 5
        self._init_dlq()

    def _model_headers_to_kafka(self, headers: object) -> list[tuple[str, bytes]]:
        return []


class _RawRecord:
    """The shape the boundary hands the raw publisher: bytes off the wire."""

    def __init__(self, value: bytes) -> None:
        self.key = b"edge-1"
        self.value = value
        self.offset = 137
        self.partition = 0
        self.headers = ()


def _attach_command_bytes() -> bytes:
    """The live record shape: an envelope whose payload holds the token."""
    return json.dumps(
        {
            "event_type": "gateway.attach",
            "correlation_id": "6f7d2a10-0000-4000-8000-000000000001",
            "payload": {
                "access_token": _SENTINEL_TOKEN,
                "edge_instance_id": _SENTINEL_SURVIVES,
            },
        }
    ).encode("utf-8")


async def _publish_raw(value: bytes, correlation_id: UUID) -> dict[str, Any]:
    host = _DlqHost()
    published = await host._publish_raw_to_dlq(
        original_topic=_ORIGINAL_TOPIC,
        raw_msg=_RawRecord(value),
        error=ValueError("keycloak introspection failed"),
        correlation_id=correlation_id,
        failure_type="handler_exception",
        consumer_group="test-group",
    )
    assert published is True
    producer: _CapturingProducer = host._producer  # type: ignore[assignment]
    assert len(producer.sent) == 1, producer.sent
    return producer.sent[0]


# ---------------------------------------------------------------------------
# AC3 -- the dead-letter publisher redacts by field name (the raw path)
# ---------------------------------------------------------------------------


async def test_raw_dlq_publish_does_not_carry_the_sentinel_token() -> None:
    """RED on origin/dev: the sentinel is published verbatim.

    This is the exact live shape of offsets 137/138.
    """
    correlation_id = uuid4()
    sent = await _publish_raw(_attach_command_bytes(), correlation_id)
    raw_bytes = sent["value"]

    assert _SENTINEL_TOKEN.encode("utf-8") not in raw_bytes, (
        "the dead-letter record published the credential verbatim; this is the "
        "defect that put two customer bearer tokens on a durable topic "
        "(onex.dlq.omnibase-infra.commands.v1 offsets 137/138, onex-dev)"
    )
    # And nowhere in the headers either.
    for _name, header_value in sent["headers"] or []:
        assert _SENTINEL_TOKEN.encode("utf-8") not in header_value


async def test_raw_dlq_publish_keeps_the_correlation_id_and_non_credentials() -> None:
    """The positive control: redaction must not blank the record.

    A dead-letter record that survives by being empty is not a fix -- it
    destroys the forensic value the topic exists for.
    """
    correlation_id = uuid4()
    sent = await _publish_raw(_attach_command_bytes(), correlation_id)
    payload = json.loads(sent["value"].decode("utf-8"))

    assert payload["correlation_id"] == str(correlation_id)
    assert payload["original_topic"] == _ORIGINAL_TOPIC
    assert payload["original_message"]["offset"] == 137
    assert payload["original_message"]["partition"] == 0

    body = json.loads(payload["original_message"]["value"])
    assert body["event_type"] == "gateway.attach"
    assert body["payload"]["edge_instance_id"] == _SENTINEL_SURVIVES, (
        "a non-credential field was destroyed; the redactor must remove the "
        "credential, not the record"
    )
    assert body["payload"]["access_token"] == DLQ_REDACTION_MARKER


async def test_raw_dlq_publish_records_which_field_names_were_redacted() -> None:
    """The envelope must say a field was removed -- names only, never values.

    Without this, a reader cannot tell a record that never carried a token
    from one whose token was stripped, and the replay engine cannot refuse to
    republish a body it knows is incomplete.
    """
    sent = await _publish_raw(_attach_command_bytes(), uuid4())
    payload = json.loads(sent["value"].decode("utf-8"))

    assert payload["redacted_fields"] == ["payload.access_token"]
    assert _SENTINEL_TOKEN not in json.dumps(payload["redacted_fields"])


async def test_raw_dlq_publish_leaves_a_clean_record_byte_identical() -> None:
    """A record with no credential field must be unchanged and unannotated."""
    clean = json.dumps({"payload": {"edge_instance_id": _SENTINEL_SURVIVES}}).encode()
    sent = await _publish_raw(clean, uuid4())
    payload = json.loads(sent["value"].decode("utf-8"))

    assert payload["original_message"]["value"] == clean.decode("utf-8")
    assert "redacted_fields" not in payload


async def test_raw_dlq_publish_redacts_a_non_json_body() -> None:
    """A form-encoded / log-shaped body still gets a best-effort scrub."""
    sent = await _publish_raw(
        f"grant_type=client_credentials&access_token={_SENTINEL_TOKEN}".encode(),
        uuid4(),
    )
    raw_bytes = sent["value"]
    assert _SENTINEL_TOKEN.encode("utf-8") not in raw_bytes
    assert b"client_credentials" in raw_bytes


async def test_raw_dlq_publish_redacts_a_future_plain_str_field() -> None:
    """Defense in depth: a field NOT covered by AC1's typing is still caught.

    This is the whole reason AC3 exists as a separate layer -- a field added
    later as a plain ``str``, or one placed in a free-form dict that no type
    can reach (``ModelLlmInferenceCommand.provider_config`` is the live
    example), must not leak.
    """
    body = json.dumps(
        {
            "provider_config": {"api_key": _SENTINEL_TOKEN},
            "nested": [{"client_secret": _SENTINEL_TOKEN}],
            "model": _SENTINEL_SURVIVES,
        }
    ).encode("utf-8")
    sent = await _publish_raw(body, uuid4())
    raw_bytes = sent["value"]

    assert _SENTINEL_TOKEN.encode("utf-8") not in raw_bytes
    assert _SENTINEL_SURVIVES.encode("utf-8") in raw_bytes
    payload = json.loads(raw_bytes.decode("utf-8"))
    assert sorted(payload["redacted_fields"]) == [
        "nested[0].client_secret",
        "provider_config.api_key",
    ]


# ---------------------------------------------------------------------------
# AC1 -- the credential fields are typed so the model cannot serialise them
# ---------------------------------------------------------------------------


def test_attach_request_access_token_is_secret_wrapped() -> None:
    """RED on origin/dev: ``access_token`` is ``str`` and dumps verbatim."""
    request = ModelGatewayAttachRequest(
        access_token=SecretStr(_SENTINEL_TOKEN),
        edge_instance_id=_SENTINEL_SURVIVES,
    )

    assert _SENTINEL_TOKEN not in request.model_dump_json()
    assert _SENTINEL_TOKEN not in json.dumps(request.model_dump(mode="json"))
    assert _SENTINEL_TOKEN not in repr(request)
    assert _SENTINEL_TOKEN not in str(request)
    # The edge id must survive all of them -- the positive control again.
    assert _SENTINEL_SURVIVES in request.model_dump_json()


def test_attach_request_handler_can_still_read_the_token() -> None:
    """Typing must not break the one consumer that needs the real value."""
    request = ModelGatewayAttachRequest(
        access_token=SecretStr(_SENTINEL_TOKEN),
        edge_instance_id=_SENTINEL_SURVIVES,
    )
    assert request.access_token.get_secret_value() == _SENTINEL_TOKEN


def test_attach_request_parses_a_plain_string_off_the_wire() -> None:
    """A bus payload carries a JSON string; it must still validate."""
    request = ModelGatewayAttachRequest.model_validate(
        {"access_token": _SENTINEL_TOKEN, "edge_instance_id": _SENTINEL_SURVIVES}
    )
    assert request.access_token.get_secret_value() == _SENTINEL_TOKEN


# ---------------------------------------------------------------------------
# The name rule itself -- over-redaction is a defect too
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "access_token",
        "accessToken",
        "ACCESS-TOKEN",
        "api_key",
        "apiKey",
        "client_secret",
        "password",
        "passphrase",
        "authorization",
        "refresh_token",
        "private_key",
        "credential",
        "token",
        "secret",
        "x_access_token",
    ],
)
def test_credential_field_names_are_redacted(name: str) -> None:
    assert is_credential_field_name(name) is True


@pytest.mark.parametrize(
    "name",
    [
        # Counts, not credentials -- the LLM models are full of these.
        "max_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "token_count",
        "token_type",
        # References to a credential, which is the platform's own pattern.
        "secret_ref",
        "api_key_ref",
        "client_secret_api_key_ref",
        "api_key_id",
        "credential_reference_id",
        "private_key_path",
        "credentials_output_path",
        # Names and scopes, not values.
        "required_secrets",
        "secret_scopes",
        "permitted_secret_scopes",
        "credentials_captured",
        # Ordinary fields that must stay readable on a dead-letter record.
        "edge_instance_id",
        "correlation_id",
        "session_id",
        "model",
        "event_type",
    ],
)
def test_non_credential_field_names_are_not_redacted(name: str) -> None:
    assert is_credential_field_name(name) is False


def test_redaction_reports_every_path_it_touched() -> None:
    value, paths = redact_credential_fields(
        {
            "a": {"access_token": _SENTINEL_TOKEN},
            "b": [{"password": _SENTINEL_TOKEN}, {"keep": _SENTINEL_SURVIVES}],
            "token_count": 7,
        }
    )
    assert sorted(paths) == ["a.access_token", "b[0].password"]
    assert value == {
        "a": {"access_token": DLQ_REDACTION_MARKER},
        "b": [{"password": DLQ_REDACTION_MARKER}, {"keep": _SENTINEL_SURVIVES}],
        "token_count": 7,
    }


def test_text_redaction_returns_the_input_unchanged_when_nothing_matched() -> None:
    text = json.dumps({"model": _SENTINEL_SURVIVES, "max_tokens": 10})
    out, paths = redact_credential_fields_in_text(text)
    assert out == text
    assert paths == ()


def test_deeply_nested_body_fails_closed_rather_than_passing_through() -> None:
    """An attacker-shaped nesting depth must not become an unredacted copy."""
    body: dict[str, object] = {"access_token": _SENTINEL_TOKEN}
    for _ in range(200):
        body = {"next": body}
    out, paths = redact_credential_fields(body)
    serialised = json.dumps(out)
    assert _SENTINEL_TOKEN not in serialised
    assert paths, "the depth cut-off must be recorded, not silent"
