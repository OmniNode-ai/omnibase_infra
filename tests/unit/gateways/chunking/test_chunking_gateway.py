# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ChunkingGateway."""

import pytest

from omnibase_core.models.chunking.model_chunk_policy import ModelChunkPolicy
from omnibase_core.models.chunking.model_chunked_envelope import ModelChunkedEnvelope
from omnibase_infra.gateways.chunking.chunking_gateway import ChunkingGateway
from omnibase_infra.gateways.chunking.default_chunker import DefaultEnvelopeChunker


class _FakeEnvelope:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def to_bytes(self) -> bytes:
        return self._payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "_FakeEnvelope":
        return cls(data)


@pytest.mark.unit
class TestChunkingGateway:
    def _make_gateway(
        self,
        max_payload_size_bytes: int = 900_000,
        chunk_target_size_bytes: int = 256_000,
        enabled: bool = True,
    ) -> ChunkingGateway:
        policy = ModelChunkPolicy(
            enabled=enabled,
            max_payload_size_bytes=max_payload_size_bytes,
            chunk_target_size_bytes=chunk_target_size_bytes,
        )
        return ChunkingGateway(chunker=DefaultEnvelopeChunker(), policy=policy)

    def test_small_payload_returns_no_chunks(self) -> None:
        """Payload below threshold → returns empty list (pass-through signal)."""
        gateway = self._make_gateway(max_payload_size_bytes=1000)
        envelope = _FakeEnvelope(b"small" * 10)
        result = gateway.evaluate_and_chunk(envelope)
        assert result == []

    def test_large_payload_returns_chunked_envelopes(self) -> None:
        """Payload above threshold → returns list of ModelChunkedEnvelope."""
        gateway = self._make_gateway(
            max_payload_size_bytes=100, chunk_target_size_bytes=50
        )
        envelope = _FakeEnvelope(b"x" * 300)
        result = gateway.evaluate_and_chunk(envelope)
        assert len(result) >= 2
        for chunk in result:
            assert isinstance(chunk, ModelChunkedEnvelope)

    def test_disabled_policy_always_returns_empty(self) -> None:
        """When chunking disabled, always pass-through regardless of payload size."""
        gateway = self._make_gateway(max_payload_size_bytes=10, enabled=False)
        envelope = _FakeEnvelope(b"very large payload that would normally chunk " * 100)
        result = gateway.evaluate_and_chunk(envelope)
        assert result == []

    def test_chunks_cover_full_payload(self) -> None:
        """Concatenated chunk payloads equal the original bytes."""
        gateway = self._make_gateway(
            max_payload_size_bytes=100, chunk_target_size_bytes=50
        )
        payload = b"full coverage test " * 20
        envelope = _FakeEnvelope(payload)
        chunks = gateway.evaluate_and_chunk(envelope)
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_metadata.chunk_index)
        reconstructed = b"".join(c.chunk_payload for c in sorted_chunks)
        assert reconstructed == payload

    def test_exactly_at_threshold_does_not_chunk(self) -> None:
        """Payload exactly equal to max_payload_size_bytes → no chunking."""
        gateway = self._make_gateway(max_payload_size_bytes=100)
        envelope = _FakeEnvelope(b"a" * 100)
        result = gateway.evaluate_and_chunk(envelope)
        assert result == []

    def test_one_byte_over_threshold_chunks(self) -> None:
        """Payload one byte over threshold → chunked."""
        gateway = self._make_gateway(
            max_payload_size_bytes=100, chunk_target_size_bytes=60
        )
        envelope = _FakeEnvelope(b"a" * 101)
        result = gateway.evaluate_and_chunk(envelope)
        assert len(result) >= 1
