# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for node_merge_gate_effect Linear config injection (OMN-10814).

Proves the contract-to-handler wiring for the LINEAR_API_KEY config injection
path: the contract.yaml declares the two config keys the DI/secret resolver is
expected to supply, and the handler consumes those injected values rather than
reading os.environ directly. The regression this guards against is the original
``os.environ.get("LINEAR_API_KEY")`` read, which had no contract declaration and
no injection path, so the resolver could never supply the credential.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.nodes.node_merge_gate_effect.handlers.handler_upsert_merge_gate import (
    HandlerUpsertMergeGate,
)
from omnibase_infra.nodes.node_merge_gate_effect.models.model_merge_gate_result import (
    ModelMergeGateResult,
    ModelMergeGateViolation,
)

pytestmark = [pytest.mark.integration]

CONTRACT = (
    Path(__file__).parents[3]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_merge_gate_effect"
    / "contract.yaml"
)


def _make_quarantine_payload() -> ModelMergeGateResult:
    from datetime import UTC, datetime

    return ModelMergeGateResult(
        gate_id=uuid4(),
        pr_ref="OmniNode-ai/omnibase_infra#42",
        head_sha="abc123def456",
        base_sha="000111222333",
        decision="QUARANTINE",
        tier="tier-a",
        violations=[
            ModelMergeGateViolation(
                rule_code="RRH-1001",
                severity="FAIL",
                message="Working tree is dirty",
            ),
        ],
        run_id=uuid4(),
        correlation_id=uuid4(),
        run_fingerprint="fp-test-12345",
        decided_at=datetime.now(tz=UTC),
    )


def _make_pool() -> MagicMock:
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"was_insert": True})
    pool = MagicMock()
    acquire = AsyncMock()
    acquire.__aenter__ = AsyncMock(return_value=conn)
    acquire.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=acquire)
    return pool


def test_contract_declares_linear_config_keys() -> None:
    """The contract declares the config keys the resolver injects into the handler."""
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    config = contract["config"]

    assert "linear_api_key" in config
    assert "linear_team_id" in config
    # Optional credentials: default null so the resolver may legitimately omit
    # them, in which case QUARANTINE ticket creation is skipped.
    assert config["linear_api_key"]["default"] is None
    assert config["linear_team_id"]["default"] is None


@pytest.mark.asyncio
async def test_injected_credential_reaches_linear_request_not_env() -> None:
    """The injected key flows to the Linear request even when env vars are unset.

    With os.environ cleared of LINEAR_API_KEY/LINEAR_TEAM_ID, the handler still
    creates the ticket using the constructor-injected credential — proving the
    DI path is the source of truth, not the removed os.environ read.
    """
    payload = _make_quarantine_payload()

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(
        return_value={
            "data": {
                "issueCreate": {
                    "success": True,
                    "issue": {
                        "id": "i1",
                        "identifier": "OMN-1",
                        "url": "https://linear.app/omninode/issue/OMN-1",
                    },
                }
            }
        }
    )

    with (
        patch.dict(
            "os.environ",
            {"LINEAR_API_KEY": "", "LINEAR_TEAM_ID": ""},
            clear=False,
        ),
        patch(
            "omnibase_infra.nodes.node_merge_gate_effect.handlers."
            "handler_upsert_merge_gate.httpx.AsyncClient"
        ) as mock_client_cls,
    ):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        handler = HandlerUpsertMergeGate(
            _make_pool(),
            linear_api_key="resolver-injected-key",
            linear_team_id="resolver-injected-team",
        )
        result = await handler.handle(payload, uuid4())

    assert result.success is True
    mock_client.post.assert_called_once()
    post_kwargs = mock_client.post.call_args[1]
    assert post_kwargs["headers"]["Authorization"] == "resolver-injected-key"
    assert post_kwargs["json"]["variables"]["teamId"] == "resolver-injected-team"


@pytest.mark.asyncio
async def test_missing_injected_credential_skips_linear_call() -> None:
    """Without injected credentials the upsert succeeds and no Linear call is made."""
    payload = _make_quarantine_payload()

    with patch(
        "omnibase_infra.nodes.node_merge_gate_effect.handlers."
        "handler_upsert_merge_gate.httpx.AsyncClient"
    ) as mock_client_cls:
        handler = HandlerUpsertMergeGate(_make_pool())
        result = await handler.handle(payload, uuid4())

    assert result.success is True
    mock_client_cls.assert_not_called()
