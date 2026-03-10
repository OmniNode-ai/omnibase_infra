# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration test: ChunkingGateway → ReassemblyGateway roundtrip."""

import pytest

from omnibase_core.models.chunking.model_chunk_policy import ModelChunkPolicy
from omnibase_core.models.chunking.model_chunk_series_failed import (
    ModelChunkSeriesFailed,
)
from omnibase_infra.gateways.chunking.chunking_gateway import ChunkingGateway
from omnibase_infra.gateways.chunking.default_chunker import DefaultEnvelopeChunker
from omnibase_infra.gateways.chunking.reassembly_gateway import ReassemblyGateway


class _RoundtripEnvelope:
    """Minimal chunkable envelope for integration testing."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def to_bytes(self) -> bytes:
        return self._payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "_RoundtripEnvelope":
        return cls(data)


@pytest.mark.integration
class TestChunkReassembleRoundtrip:
    def _make_pipeline(
        self,
        max_payload_size_bytes: int,
        chunk_target_size_bytes: int,
    ) -> tuple[ChunkingGateway, ReassemblyGateway]:
        chunker = DefaultEnvelopeChunker()
        policy = ModelChunkPolicy(
            enabled=True,
            max_payload_size_bytes=max_payload_size_bytes,
            chunk_target_size_bytes=chunk_target_size_bytes,
        )
        return ChunkingGateway(chunker=chunker, policy=policy), ReassemblyGateway(
            chunker=chunker
        )

    def test_large_envelope_roundtrip(self) -> None:
        """Large envelope: chunk then reassemble yields identical bytes."""
        payload = b"roundtrip payload data " * 500
        chunking_gw, reassembly_gw = self._make_pipeline(
            max_payload_size_bytes=1000, chunk_target_size_bytes=400
        )
        envelope = _RoundtripEnvelope(payload)
        chunks = chunking_gw.evaluate_and_chunk(envelope)
        assert len(chunks) > 1

        result = None
        for chunk in chunks:
            result = reassembly_gw.receive(chunk, _RoundtripEnvelope)

        assert result is not None
        assert not isinstance(result, ModelChunkSeriesFailed)
        assert result.to_bytes() == payload

    def test_small_envelope_passthrough(self) -> None:
        """Small envelope below threshold is not chunked (empty list from gateway)."""
        payload = b"tiny payload"
        chunking_gw, _reassembly_gw = self._make_pipeline(
            max_payload_size_bytes=10_000, chunk_target_size_bytes=5_000
        )
        envelope = _RoundtripEnvelope(payload)
        chunks = chunking_gw.evaluate_and_chunk(envelope)
        assert chunks == []

    def test_exactly_two_chunks_roundtrip(self) -> None:
        """Payload that splits into exactly 2 chunks reassembles correctly."""
        payload = b"half and half " * 100
        chunking_gw, reassembly_gw = self._make_pipeline(
            max_payload_size_bytes=len(payload) - 1,
            chunk_target_size_bytes=len(payload) // 2,
        )
        envelope = _RoundtripEnvelope(payload)
        chunks = chunking_gw.evaluate_and_chunk(envelope)
        assert len(chunks) == 2

        result = None
        for chunk in chunks:
            result = reassembly_gw.receive(chunk, _RoundtripEnvelope)

        assert result is not None
        assert not isinstance(result, ModelChunkSeriesFailed)
        assert result.to_bytes() == payload

    def test_many_small_chunks_roundtrip(self) -> None:
        """Many small chunks (stress) → full roundtrip."""
        payload = bytes(range(256)) * 40  # 10240 bytes
        chunking_gw, reassembly_gw = self._make_pipeline(
            max_payload_size_bytes=500, chunk_target_size_bytes=100
        )
        envelope = _RoundtripEnvelope(payload)
        chunks = chunking_gw.evaluate_and_chunk(envelope)
        assert len(chunks) > 10

        result = None
        for chunk in chunks:
            result = reassembly_gw.receive(chunk, _RoundtripEnvelope)

        assert result is not None
        assert not isinstance(result, ModelChunkSeriesFailed)
        assert result.to_bytes() == payload
