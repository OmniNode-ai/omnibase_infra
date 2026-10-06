# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A 402 from an LLM backend is a typed answer that ends the attempt (OMN-20608, T0.1).

One HTTP attempt, no retry, no circuit breaker failure, and the sha256 and byte
length of the full raw header and body are taken before any truncation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Generator
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.errors import (
    InfraPaymentRequiredError,
    InfraRequestRejectedError,
    InfraUnavailableError,
)
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport

URL = "http://192.168.86.201:8000/v1/chat/completions"
PAYLOAD = {"messages": [{"role": "user", "content": "hello"}]}
HEADER_NAME = "payment-required"


class _Harness(MixinLlmHttpTransport):
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._init_llm_http_transport(
            target_name="test-llm",
            max_timeout_seconds=120.0,
            max_retry_after_seconds=30.0,
            http_client=client,
        )


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("LOCAL_LLM_SHARED_SECRET", "t" * 32)
    monkeypatch.setenv("LLM_ENDPOINT_CIDR_ALLOWLIST", "192.168.86.0/24")
    monkeypatch.setenv("LLM_CLOUD_ENDPOINT_HOST_ALLOWLIST", "api.z.ai")
    for cls in (MixinLlmHttpTransport, _Harness):
        cls._LOCAL_LLM_CIDRS = None
        cls._CLOUD_LLM_HOSTS = None
    yield
    for cls in (MixinLlmHttpTransport, _Harness):
        cls._LOCAL_LLM_CIDRS = None
        cls._CLOUD_LLM_HOSTS = None


def _harness(
    status: int, headers: dict[str, str], body: bytes
) -> tuple[_Harness, list[int]]:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status, headers=headers, content=body)

    return _Harness(httpx.AsyncClient(transport=httpx.MockTransport(handler))), calls


@pytest.mark.unit
class TestPaymentRequired:
    async def test_402_is_one_attempt_with_no_breaker_failure(self) -> None:
        harness, calls = _harness(
            402, {"content-type": "application/json", HEADER_NAME: "abc"}, b"{}"
        )
        before = harness._circuit_breaker_failures
        with pytest.raises(InfraPaymentRequiredError) as exc_info:
            await harness._execute_llm_http_call(
                url=URL, payload=PAYLOAD, correlation_id=uuid4(), max_retries=3
            )
        assert len(calls) == 1
        assert harness._circuit_breaker_failures == before
        assert exc_info.value.status_code == 402
        assert isinstance(exc_info.value, InfraRequestRejectedError)

    async def test_digest_covers_full_header_and_body_before_truncation(self) -> None:
        header = "H" * (20 * 1024)
        body = b'{"pay_to":"' + b"B" * 5000 + b'"}'
        harness, _ = _harness(
            402, {"content-type": "application/json", HEADER_NAME: header}, body
        )
        with pytest.raises(InfraPaymentRequiredError) as exc_info:
            await harness._execute_llm_http_call(
                url=URL, payload=PAYLOAD, correlation_id=uuid4(), max_retries=3
            )
        err = exc_info.value
        assert err.header_sha256 == hashlib.sha256(header.encode()).hexdigest()
        assert err.header_byte_length == 20 * 1024
        assert err.body_sha256 == hashlib.sha256(body).hexdigest()
        assert err.body_byte_length == len(body)
        assert len(err.payment_required_header) == 8 * 1024
        assert len(err.response_body) < len(body)

    async def test_missing_header_has_empty_digest_fields(self) -> None:
        harness, _ = _harness(402, {"content-type": "application/json"}, b"{}")
        with pytest.raises(InfraPaymentRequiredError) as exc_info:
            await harness._execute_llm_http_call(
                url=URL, payload=PAYLOAD, correlation_id=uuid4(), max_retries=0
            )
        assert exc_info.value.payment_required_header == ""
        assert exc_info.value.header_sha256 == ""
        assert exc_info.value.header_byte_length == 0

    async def test_classification_is_terminal_and_not_a_breaker_failure(self) -> None:
        harness, _ = _harness(402, {}, b"")
        error = InfraPaymentRequiredError("402", status_code=402)
        classification = harness._classify_error(error, "op")
        assert classification.should_retry is False
        assert classification.record_circuit_failure is False

    async def test_500_still_retries_and_counts_against_the_breaker(self) -> None:
        harness, calls = _harness(500, {"content-type": "application/json"}, b"{}")
        with pytest.raises(InfraUnavailableError):
            await harness._execute_llm_http_call(
                url=URL, payload=PAYLOAD, correlation_id=uuid4(), max_retries=1
            )
        assert len(calls) == 2
        assert harness._circuit_breaker_failures >= 1
