# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerSourceAttestation — OMN-9139.

Tests the core attestation logic without hitting the network.
All git subprocess calls are patched out.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.health.model_runtime_booted_event import (
    ModelRuntimeBootedEvent,
)
from omnibase_infra.nodes.node_runtime_source_attestor_effect.handlers.handler_source_attestation import (
    HandlerSourceAttestation,
    ModelSourceAttestationResult,
)

pytestmark = [pytest.mark.unit]

_FAKE_MAIN_HEAD = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
_FAKE_HASH_AT_HEAD = "a1b2c3d"  # short prefix of _FAKE_MAIN_HEAD
_FAKE_OLD_HASH = "deadbeef1234567"


def _make_event(
    runtime_source_hash: str = _FAKE_HASH_AT_HEAD,
    container_ref: str = "omnibase-infra-omninode-runtime",
) -> ModelRuntimeBootedEvent:
    return ModelRuntimeBootedEvent(
        container_ref=container_ref,
        runtime_source_hash=runtime_source_hash,
        booted_at=datetime.now(UTC),
        python_package_hashes={},
    )


def _make_handler(tmp_path: Path, drift_threshold: int = 5) -> HandlerSourceAttestation:
    """Return a handler wired to a temp friction dir with no real git calls."""
    return HandlerSourceAttestation(
        repo_url="https://example.com/fake.git",
        drift_threshold=drift_threshold,
        friction_dir=tmp_path / "friction",
    )


# ---------------------------------------------------------------------------
# ModelRuntimeBootedEvent — model construction
# ---------------------------------------------------------------------------


class TestModelRuntimeBootedEvent:
    def test_valid_construction(self) -> None:
        event = _make_event()
        assert event.container_ref == "omnibase-infra-omninode-runtime"
        assert event.runtime_source_hash == _FAKE_HASH_AT_HEAD
        assert isinstance(event.booted_at, datetime)

    def test_default_compose_project(self) -> None:
        event = _make_event()
        assert event.compose_project == "unknown"

    def test_frozen(self) -> None:
        event = _make_event()
        with pytest.raises(Exception):
            event.container_ref = "new-name"  # type: ignore[misc]

    def test_extra_fields_ignored_for_forward_compatibility(self) -> None:
        event = ModelRuntimeBootedEvent(
            container_ref="c",
            runtime_source_hash="abc1234",
            booted_at=datetime.now(UTC),
            unexpected_field="boom",  # type: ignore[call-arg]
        )

        assert event.container_ref == "c"
        assert not hasattr(event, "unexpected_field")


# ---------------------------------------------------------------------------
# ModelSourceAttestationResult — model construction
# ---------------------------------------------------------------------------


class TestModelSourceAttestationResult:
    def test_valid_construction(self) -> None:
        result = ModelSourceAttestationResult(
            container_ref="test-container",
            runtime_source_hash="abc1234",
            verdict="compliant",
        )
        assert result.verdict == "compliant"
        assert result.commit_distance == -1
        assert result.friction_path is None

    def test_frozen(self) -> None:
        result = ModelSourceAttestationResult(
            container_ref="c",
            runtime_source_hash="abc1234",
            verdict="compliant",
        )
        with pytest.raises(Exception):
            result.verdict = "drifted"  # type: ignore[misc]


class TestHandlerMetadata:
    def test_handler_classification_properties(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)

        assert handler.handler_type is EnumHandlerType.NODE_HANDLER
        assert handler.handler_category is EnumHandlerTypeCategory.EFFECT


# ---------------------------------------------------------------------------
# unknown / empty hash → always friction
# ---------------------------------------------------------------------------


class TestUnknownHash:
    @pytest.mark.parametrize("bad_hash", ["unknown", "", "dev", "  "])
    def test_unknown_hash_emits_friction(self, bad_hash: str, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        event = _make_event(runtime_source_hash=bad_hash)

        with patch.object(handler, "_resolve_main_head", return_value=None):
            result = handler.attest(event)

        assert result.verdict == "unknown_hash"
        assert result.friction_path is not None
        friction_file = Path(result.friction_path)
        assert friction_file.exists()
        data = yaml.safe_load(friction_file.read_text())
        assert data["type"] == "runtime_source_drift"
        assert data["ticket"] == "OMN-9139"

    def test_unknown_hash_does_not_need_network(self, tmp_path: Path) -> None:
        """_resolve_main_head must NOT be called for unknown hashes."""
        handler = _make_handler(tmp_path)
        event = _make_event(runtime_source_hash="unknown")

        with patch.object(
            handler, "_resolve_main_head", side_effect=AssertionError("should not call")
        ):
            # The handler short-circuits before calling _resolve_main_head
            result = handler.attest(event)

        assert result.verdict == "unknown_hash"


# ---------------------------------------------------------------------------
# hash matches HEAD → compliant
# ---------------------------------------------------------------------------


class TestCompliantHash:
    def test_exact_short_hash_match_is_compliant(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        # runtime hash is a 7-char prefix of main HEAD
        event = _make_event(runtime_source_hash=_FAKE_MAIN_HEAD[:7])

        with patch.object(handler, "_resolve_main_head", return_value=_FAKE_MAIN_HEAD):
            with patch.object(handler, "_compute_distance", return_value=0):
                result = handler.attest(event)

        assert result.verdict == "compliant"
        assert result.friction_path is None

    def test_full_sha_match_is_compliant(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        event = _make_event(runtime_source_hash=_FAKE_MAIN_HEAD)

        with patch.object(handler, "_resolve_main_head", return_value=_FAKE_MAIN_HEAD):
            with patch.object(handler, "_compute_distance", return_value=0):
                result = handler.attest(event)

        assert result.verdict == "compliant"

    def test_within_drift_threshold_is_compliant(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path, drift_threshold=5)
        event = _make_event(runtime_source_hash=_FAKE_OLD_HASH)

        with patch.object(handler, "_resolve_main_head", return_value=_FAKE_MAIN_HEAD):
            with patch.object(handler, "_compute_distance", return_value=3):
                result = handler.attest(event)

        assert result.verdict == "compliant"
        assert result.commit_distance == 3
        assert result.friction_path is None

    def test_shared_short_prefix_is_not_exact_match(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path, drift_threshold=5)
        runtime_hash = f"{_FAKE_MAIN_HEAD[:7]}999999999999999999999999999999999"
        event = _make_event(runtime_source_hash=runtime_hash)

        with patch.object(handler, "_resolve_main_head", return_value=_FAKE_MAIN_HEAD):
            with patch.object(handler, "_compute_distance", return_value=-1):
                result = handler.attest(event)

        assert result.verdict == "drifted"
        assert result.friction_path is not None


# ---------------------------------------------------------------------------
# hash is stale → drifted + friction
# ---------------------------------------------------------------------------


class TestDriftedHash:
    def test_over_threshold_emits_friction(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path, drift_threshold=5)
        event = _make_event(runtime_source_hash=_FAKE_OLD_HASH)

        with patch.object(handler, "_resolve_main_head", return_value=_FAKE_MAIN_HEAD):
            with patch.object(handler, "_compute_distance", return_value=10):
                result = handler.attest(event)

        assert result.verdict == "drifted"
        assert result.commit_distance == 10
        assert result.friction_path is not None
        friction_file = Path(result.friction_path)
        assert friction_file.exists()
        data = yaml.safe_load(friction_file.read_text())
        assert data["type"] == "runtime_source_drift"
        assert data["commit_distance"] == 10

    def test_distance_unknown_and_no_match_is_drifted(self, tmp_path: Path) -> None:
        """When git ls-remote works but rev-list fails (-1), non-matching hash → drifted."""
        handler = _make_handler(tmp_path, drift_threshold=5)
        event = _make_event(runtime_source_hash=_FAKE_OLD_HASH)

        with patch.object(handler, "_resolve_main_head", return_value=_FAKE_MAIN_HEAD):
            with patch.object(handler, "_compute_distance", return_value=-1):
                result = handler.attest(event)

        assert result.verdict == "drifted"
        assert result.commit_distance == -1
        assert result.friction_path is not None

    def test_friction_file_slug_escapes_slashes(self, tmp_path: Path) -> None:
        """Container names with slashes produce valid filenames."""
        handler = _make_handler(tmp_path, drift_threshold=0)
        event = _make_event(
            runtime_source_hash=_FAKE_OLD_HASH,
            container_ref="org/project:latest",
        )

        with patch.object(handler, "_resolve_main_head", return_value=_FAKE_MAIN_HEAD):
            with patch.object(handler, "_compute_distance", return_value=99):
                result = handler.attest(event)

        assert result.friction_path is not None
        assert "/" not in Path(result.friction_path).name
        assert ":" not in Path(result.friction_path).name

    def test_git_ls_remote_unavailable_and_no_match_is_drifted(
        self, tmp_path: Path
    ) -> None:
        """When git ls-remote is completely unavailable (returns None), non-matching hash → drifted."""
        handler = _make_handler(tmp_path, drift_threshold=5)
        event = _make_event(runtime_source_hash=_FAKE_OLD_HASH)

        with patch.object(handler, "_resolve_main_head", return_value=None):
            result = handler.attest(event)

        assert result.verdict == "drifted"
        assert result.friction_path is not None


# ---------------------------------------------------------------------------
# _resolve_main_head error handling
# ---------------------------------------------------------------------------


class TestResolveMainHead:
    def test_returns_none_on_git_failure(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = handler._resolve_main_head()

        assert result is None

    def test_returns_sha_on_success(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = f"{_FAKE_MAIN_HEAD}\trefs/heads/main\n"

        with patch("subprocess.run", return_value=mock_result):
            result = handler._resolve_main_head()

        assert result == _FAKE_MAIN_HEAD

    def test_returns_none_on_exception(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        with patch("subprocess.run", side_effect=OSError("git not found")):
            result = handler._resolve_main_head()

        assert result is None


class TestComputeDistance:
    def test_shared_short_prefix_is_not_distance_zero(self, tmp_path: Path) -> None:
        handler = _make_handler(tmp_path)
        runtime_hash = f"{_FAKE_MAIN_HEAD[:7]}999999999999999999999999999999999"
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = handler._compute_distance(runtime_hash, _FAKE_MAIN_HEAD)

        assert result == -1


# ---------------------------------------------------------------------------
# Dockerfile attestation contract (static analysis — no Docker build needed)
# ---------------------------------------------------------------------------


class TestDockerfileAttestationContract:
    """Verify Dockerfile.runtime declares RUNTIME_SOURCE_HASH as ARG+ENV.

    These are static checks on the Dockerfile text — no Docker daemon required.
    They enforce the build-time attestation contract from OMN-9139 DoD item 1.
    """

    @pytest.fixture(scope="class")
    def dockerfile_content(self) -> str:
        project_root = Path(__file__).parent.parent.parent.parent.parent
        dockerfile = project_root / "docker" / "Dockerfile.runtime"
        assert dockerfile.exists(), f"Dockerfile.runtime not found at {dockerfile}"
        return dockerfile.read_text()

    def test_runtime_source_hash_arg_declared(self, dockerfile_content: str) -> None:
        """RUNTIME_SOURCE_HASH must be declared as ARG in the runtime stage."""
        assert "ARG RUNTIME_SOURCE_HASH" in dockerfile_content, (
            "Dockerfile.runtime must declare 'ARG RUNTIME_SOURCE_HASH' "
            "so docker build --build-arg stamps the hash at build time (OMN-9139)."
        )

    def test_runtime_source_hash_env_set(self, dockerfile_content: str) -> None:
        """RUNTIME_SOURCE_HASH must be propagated to ENV so containers can read it."""
        assert "ENV RUNTIME_SOURCE_HASH" in dockerfile_content or (
            "RUNTIME_SOURCE_HASH=${RUNTIME_SOURCE_HASH}" in dockerfile_content
        ), (
            "Dockerfile.runtime must set ENV RUNTIME_SOURCE_HASH=... "
            "so the running container carries provenance (OMN-9139)."
        )

    def test_runtime_source_hash_default_is_unknown(
        self, dockerfile_content: str
    ) -> None:
        """Default must be 'unknown' so manual builds without the arg still start,
        but will be flagged by runtime_sweep (OMN-9122)."""
        assert "ARG RUNTIME_SOURCE_HASH=unknown" in dockerfile_content, (
            "Default value for RUNTIME_SOURCE_HASH must be 'unknown' "
            "so missing-arg builds are detectable at runtime (OMN-9139)."
        )


# ---------------------------------------------------------------------------
# docker-build.yml passes RUNTIME_SOURCE_HASH
# ---------------------------------------------------------------------------


class TestDockerBuildWorkflowAttestation:
    """Static check that docker-build.yml passes RUNTIME_SOURCE_HASH as a build-arg."""

    @pytest.fixture(scope="class")
    def workflow_content(self) -> str:
        project_root = Path(__file__).parent.parent.parent.parent.parent
        workflow = project_root / ".github" / "workflows" / "docker-build.yml"
        assert workflow.exists(), f"docker-build.yml not found at {workflow}"
        return workflow.read_text()

    def test_runtime_source_hash_in_build_args(self, workflow_content: str) -> None:
        """docker-build.yml must pass RUNTIME_SOURCE_HASH in its build-args block."""
        assert "RUNTIME_SOURCE_HASH" in workflow_content, (
            "docker-build.yml must pass RUNTIME_SOURCE_HASH as a build-arg "
            "so every PR build stamps the hash (OMN-9139)."
        )

    def test_attest_source_hash_job_present(self, workflow_content: str) -> None:
        """docker-build.yml must include the attest-source-hash job."""
        assert "attest-source-hash" in workflow_content, (
            "docker-build.yml must call the attest-source-hash reusable workflow "
            "as a required status check (OMN-9139)."
        )


# ---------------------------------------------------------------------------
# __about__.py source hash module
# ---------------------------------------------------------------------------


class TestAboutModule:
    def test_about_module_importable(self) -> None:
        from omnibase_infra import __about__

        assert hasattr(__about__, "__source_hash__")

    def test_source_hash_is_string(self) -> None:
        from omnibase_infra import __about__

        assert isinstance(__about__.__source_hash__, str)
        assert len(__about__.__source_hash__) >= 3, (
            "__source_hash__ must be at least 3 chars (e.g. 'dev' or a git SHA)"
        )
