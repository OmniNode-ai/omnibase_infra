# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for the RRH storage effect node.

Covers:
- Artifact JSON writing
- Symlink creation and update (latest_by_ticket, latest_by_repo)
- Error handling on write failure
- Contract.yaml validation
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.diagnostics.enum_verdict import EnumVerdict
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.rrh.model_rrh_result import ModelRRHResult
from omnibase_infra.nodes.architecture_validator.models.model_rule_check_result import (
    ModelRuleCheckResult,
)
from omnibase_infra.nodes.node_rrh_storage_effect.handlers.handler_rrh_storage_write import (
    HandlerRRHStorageWrite,
)
from omnibase_infra.nodes.node_rrh_storage_effect.models.model_rrh_storage_request import (
    ModelRRHStorageRequest,
)
from omnibase_infra.nodes.node_rrh_storage_effect.models.model_rrh_storage_result import (
    ModelRRHStorageResult,
)
from omnibase_infra.nodes.node_rrh_storage_effect.node import NodeRRHStorageEffect

pytestmark = [pytest.mark.unit]

CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_rrh_storage_effect"
    / "contract.yaml"
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def handler() -> HandlerRRHStorageWrite:
    return HandlerRRHStorageWrite()


@pytest.fixture
def sample_result() -> ModelRRHResult:
    return ModelRRHResult(
        checks=(
            ModelRuleCheckResult(passed=True, rule_id="RRH-1001"),
            ModelRuleCheckResult(
                passed=True, rule_id="RRH-1002", skipped=True, reason="N/A"
            ),
        ),
        verdict=EnumVerdict.PASS,
        profile_name="default",
        ticket_id="OMN-2136",
        repo_name="omnibase_infra2",
        correlation_id=uuid4(),
        evaluated_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------


class TestContractValidation:
    @pytest.fixture(scope="class")
    def contract_data(self) -> dict:
        with CONTRACT_PATH.open() as f:
            data: dict = yaml.safe_load(f)
        return data

    def test_node_type_is_effect(self, contract_data: dict) -> None:
        assert contract_data.get("node_type") == "EFFECT_GENERIC"


# ---------------------------------------------------------------
# Node declarative check
# ---------------------------------------------------------------


class TestNodeDeclarative:
    def test_no_custom_methods(self) -> None:
        custom = [
            m
            for m in dir(NodeRRHStorageEffect)
            if not m.startswith("_") and m not in dir(NodeRRHStorageEffect.__bases__[0])
        ]
        assert custom == [], f"Node has custom methods: {custom}"


# ---------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------


class TestArtifactWriting:
    @pytest.mark.anyio
    async def test_writes_json_artifact(
        self,
        handler: HandlerRRHStorageWrite,
        sample_result: ModelRRHResult,
        tmp_path: Path,
    ) -> None:
        request = ModelRRHStorageRequest(result=sample_result, output_dir=str(tmp_path))
        result = await handler.handle(request)
        assert isinstance(result, ModelRRHStorageResult)
        assert result.success is True
        assert result.artifact_path

        # Verify artifact is valid JSON.
        artifact = Path(result.artifact_path)
        assert artifact.exists()
        data = json.loads(artifact.read_text())
        assert data["verdict"] == "pass"
        assert len(data["checks"]) == 2

    @pytest.mark.anyio
    async def test_creates_directory_structure(
        self,
        handler: HandlerRRHStorageWrite,
        sample_result: ModelRRHResult,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "rrh_output"
        request = ModelRRHStorageRequest(
            result=sample_result, output_dir=str(output_dir)
        )
        result = await handler.handle(request)
        assert result.success
        assert (output_dir / "artifacts").is_dir()


# ---------------------------------------------------------------
# Symlinks
# ---------------------------------------------------------------


class TestSymlinks:
    @pytest.mark.anyio
    async def test_creates_ticket_symlink(
        self,
        handler: HandlerRRHStorageWrite,
        sample_result: ModelRRHResult,
        tmp_path: Path,
    ) -> None:
        request = ModelRRHStorageRequest(result=sample_result, output_dir=str(tmp_path))
        result = await handler.handle(request)
        assert result.ticket_symlink
        symlink = Path(result.ticket_symlink)
        assert symlink.is_symlink()
        # Resolve should point to the artifact.
        assert symlink.resolve().name.startswith("rrh_")

    @pytest.mark.anyio
    async def test_creates_repo_symlink(
        self,
        handler: HandlerRRHStorageWrite,
        sample_result: ModelRRHResult,
        tmp_path: Path,
    ) -> None:
        request = ModelRRHStorageRequest(result=sample_result, output_dir=str(tmp_path))
        result = await handler.handle(request)
        assert result.repo_symlink
        symlink = Path(result.repo_symlink)
        assert symlink.is_symlink()

    @pytest.mark.anyio
    async def test_updates_symlink_on_second_write(
        self,
        handler: HandlerRRHStorageWrite,
        sample_result: ModelRRHResult,
        tmp_path: Path,
    ) -> None:
        request = ModelRRHStorageRequest(result=sample_result, output_dir=str(tmp_path))
        result1 = await handler.handle(request)

        # Write a second result.
        result2_obj = sample_result.model_copy(update={"correlation_id": uuid4()})
        request2 = ModelRRHStorageRequest(result=result2_obj, output_dir=str(tmp_path))
        result2 = await handler.handle(request2)

        # Symlink should point to the newer artifact.
        ticket_link = Path(result2.ticket_symlink)
        assert ticket_link.is_symlink()
        assert ticket_link.resolve().name != Path(result1.artifact_path).name

    @pytest.mark.anyio
    async def test_no_symlink_without_ticket_id(
        self,
        handler: HandlerRRHStorageWrite,
        tmp_path: Path,
    ) -> None:
        result = ModelRRHResult(
            checks=(ModelRuleCheckResult(passed=True, rule_id="RRH-1001"),),
            verdict=EnumVerdict.PASS,
            profile_name="default",
            ticket_id="",  # No ticket.
            repo_name="",  # No repo.
            evaluated_at=datetime.now(UTC),
        )
        request = ModelRRHStorageRequest(result=result, output_dir=str(tmp_path))
        storage_result = await handler.handle(request)
        assert storage_result.success
        assert storage_result.ticket_symlink == ""
        assert storage_result.repo_symlink == ""


# ---------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.anyio
    async def test_rejects_relative_output_dir(
        self,
        handler: HandlerRRHStorageWrite,
        sample_result: ModelRRHResult,
    ) -> None:
        request = ModelRRHStorageRequest(
            result=sample_result, output_dir="relative/path"
        )
        result = await handler.handle(request)
        assert result.success is False
        assert "absolute" in result.error.lower()

    @pytest.mark.anyio
    async def test_rejects_dotdot_in_output_dir(
        self,
        handler: HandlerRRHStorageWrite,
        sample_result: ModelRRHResult,
    ) -> None:
        request = ModelRRHStorageRequest(
            result=sample_result, output_dir="/safe/rrh/../escape"
        )
        result = await handler.handle(request)
        assert result.success is False
        assert ".." in result.error

    @pytest.mark.anyio
    async def test_handles_write_failure(
        self,
        handler: HandlerRRHStorageWrite,
        sample_result: ModelRRHResult,
    ) -> None:
        request = ModelRRHStorageRequest(
            result=sample_result, output_dir="/nonexistent/readonly/path"
        )
        result = await handler.handle(request)
        assert result.success is False
        assert result.error


# ---------------------------------------------------------------
# Path validation guard tests (ProtocolConfigurationError)
# ---------------------------------------------------------------


class TestPathValidationGuards:
    """Verify path validation raises ProtocolConfigurationError (not ValueError).

    The handler catches all exceptions in its ``except Exception`` block to
    return a graceful result.  We patch ``sanitize_error_message`` to re-raise
    the original exception so we can assert on its type directly.
    """

    @pytest.mark.anyio
    async def test_relative_path_raises_protocol_configuration_error(
        self,
        handler: HandlerRRHStorageWrite,
        sample_result: ModelRRHResult,
    ) -> None:
        request = ModelRRHStorageRequest(
            result=sample_result, output_dir="relative/path"
        )

        def _reraise(exc: BaseException) -> str:
            raise exc

        with (
            patch(
                "omnibase_infra.nodes.node_rrh_storage_effect.handlers"
                ".handler_rrh_storage_write.sanitize_error_message",
                side_effect=_reraise,
            ),
            pytest.raises(
                ProtocolConfigurationError, match="output_dir must be absolute"
            ),
        ):
            await handler.handle(request)

    @pytest.mark.anyio
    async def test_path_traversal_raises_protocol_configuration_error(
        self,
        handler: HandlerRRHStorageWrite,
        sample_result: ModelRRHResult,
    ) -> None:
        request = ModelRRHStorageRequest(
            result=sample_result, output_dir="/safe/rrh/../escape"
        )

        def _reraise(exc: BaseException) -> str:
            raise exc

        with (
            patch(
                "omnibase_infra.nodes.node_rrh_storage_effect.handlers"
                ".handler_rrh_storage_write.sanitize_error_message",
                side_effect=_reraise,
            ),
            pytest.raises(
                ProtocolConfigurationError, match="must not contain '\\.\\.'"
            ),
        ):
            await handler.handle(request)

    @pytest.mark.anyio
    async def test_relative_path_error_includes_correlation_id(
        self,
        handler: HandlerRRHStorageWrite,
        sample_result: ModelRRHResult,
    ) -> None:
        cid = uuid4()
        request = ModelRRHStorageRequest(
            result=sample_result, output_dir="relative/path", correlation_id=cid
        )

        captured: list[BaseException] = []

        def _capture(exc: BaseException) -> str:
            captured.append(exc)
            raise exc

        with (
            patch(
                "omnibase_infra.nodes.node_rrh_storage_effect.handlers"
                ".handler_rrh_storage_write.sanitize_error_message",
                side_effect=_capture,
            ),
            pytest.raises(ProtocolConfigurationError),
        ):
            await handler.handle(request)

        assert len(captured) == 1
        err = captured[0]
        assert isinstance(err, ProtocolConfigurationError)
        # The correlation_id should be propagated from the request.
        assert err.model.correlation_id == cid

    @pytest.mark.anyio
    async def test_path_traversal_error_includes_correlation_id(
        self,
        handler: HandlerRRHStorageWrite,
        sample_result: ModelRRHResult,
    ) -> None:
        cid = uuid4()
        request = ModelRRHStorageRequest(
            result=sample_result,
            output_dir="/safe/rrh/../escape",
            correlation_id=cid,
        )

        captured: list[BaseException] = []

        def _capture(exc: BaseException) -> str:
            captured.append(exc)
            raise exc

        with (
            patch(
                "omnibase_infra.nodes.node_rrh_storage_effect.handlers"
                ".handler_rrh_storage_write.sanitize_error_message",
                side_effect=_capture,
            ),
            pytest.raises(ProtocolConfigurationError),
        ):
            await handler.handle(request)

        assert len(captured) == 1
        err = captured[0]
        assert isinstance(err, ProtocolConfigurationError)
        assert err.model.correlation_id == cid
