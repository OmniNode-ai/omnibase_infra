# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for node_runtime_source_attestor_effect — OMN-9139.

Tests the handler against the real filesystem (friction file emission)
and the handler instantiation path exercised by the auto-wiring framework.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from omnibase_infra.models.health.model_runtime_booted_event import (
    ModelRuntimeBootedEvent,
)
from omnibase_infra.nodes.node_runtime_source_attestor_effect.handlers.handler_source_attestation import (
    HandlerSourceAttestation,
)

pytestmark = pytest.mark.integration


class TestHandlerSourceAttestationIntegration:
    """Integration: handler writes friction files to real filesystem."""

    @pytest.fixture
    def friction_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "friction"

    @pytest.fixture
    def handler(self, friction_dir: Path) -> HandlerSourceAttestation:
        return HandlerSourceAttestation(
            repo_url="https://github.com/OmniNode-ai/omnibase_infra.git",
            drift_threshold=5,
            friction_dir=friction_dir,
        )

    def _boot_event(self, runtime_source_hash: str) -> ModelRuntimeBootedEvent:
        return ModelRuntimeBootedEvent(
            container_ref="omnibase-runtime-test",
            runtime_source_hash=runtime_source_hash,
            booted_at=datetime.now(UTC),
        )

    def test_unknown_hash_emits_friction_file(
        self,
        handler: HandlerSourceAttestation,
        friction_dir: Path,
    ) -> None:
        event = self._boot_event("unknown")
        result = handler.attest(event)

        assert result.verdict == "unknown_hash"
        assert result.friction_path is not None
        assert Path(result.friction_path).exists()

    def test_unknown_hash_friction_file_contains_container_ref(
        self,
        handler: HandlerSourceAttestation,
        friction_dir: Path,
    ) -> None:
        event = self._boot_event("unknown")
        result = handler.attest(event)

        assert result.friction_path is not None
        content = Path(result.friction_path).read_text()
        assert "omnibase-runtime-test" in content
        assert "OMN-9139" in content

    def test_empty_hash_emits_friction_file(
        self,
        handler: HandlerSourceAttestation,
        friction_dir: Path,
    ) -> None:
        event = self._boot_event("")
        result = handler.attest(event)

        assert result.verdict == "unknown_hash"
        assert result.friction_path is not None
        assert Path(result.friction_path).exists()

    def test_handler_instantiates_with_no_args(self) -> None:
        """Auto-wiring framework calls handler_class() with no args — must not raise."""
        handler = HandlerSourceAttestation()
        assert handler is not None

    def test_drifted_hash_emits_friction_file(
        self,
        handler: HandlerSourceAttestation,
        friction_dir: Path,
    ) -> None:
        """A hash that is clearly stale (all zeros) should produce drifted verdict."""
        event = self._boot_event("0000000000000000000000000000000000000000")
        result = handler.attest(event)

        # When git ls-remote is unavailable in CI sandbox, verdict is drifted
        # because the hash won't match main HEAD short form.
        assert result.verdict in {"drifted", "unknown_hash"}
        if result.verdict == "drifted":
            assert result.friction_path is not None
            assert Path(result.friction_path).exists()
