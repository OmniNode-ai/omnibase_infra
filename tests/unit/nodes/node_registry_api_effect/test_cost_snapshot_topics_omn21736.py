# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The cost API snapshot topics come from the registry API node contract (OMN-21736).

Verifies:
1. ``ModelCostSnapshotTopics.from_contract`` reads ``config.cost_snapshot_topics``.
2. A contract missing the section fails loudly instead of defaulting.
3. ``create_app`` binds the contract topics; a route reads an injected override.
4. The cost handlers read the snapshot under the injected topic, not a code constant.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml
from fastapi import FastAPI
from pydantic import ValidationError

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.models.model_cost_snapshot_topics import ModelCostSnapshotTopics
from omnibase_infra.nodes.node_registry_api_effect import load_registry_api_config
from omnibase_infra.services.cost_api import handlers as cost_handlers
from omnibase_infra.services.cost_api.router import (
    get_cost_snapshot_topics,
    router,
)
from omnibase_infra.services.cost_api.snapshot_cache import (
    clear_latest_snapshots,
    store_latest_snapshot,
)
from omnibase_infra.services.registry_api.main import create_app

_CONTRACT = (
    Path(__file__).resolve().parents[4]
    / "src/omnibase_infra/nodes/node_registry_api_effect/contract.yaml"
)


class _UnusedPool:
    """A pool the snapshot path must never touch."""

    def acquire(self) -> object:
        raise AssertionError("snapshot path must not query the database")


@pytest.fixture(autouse=True)
def _fresh_state() -> None:
    load_registry_api_config.cache_clear()
    clear_latest_snapshots()


def _injected_topics() -> ModelCostSnapshotTopics:
    return ModelCostSnapshotTopics(
        summary="onex.snapshot.injected.cost-summary.v1",
        by_repo="onex.snapshot.injected.cost-by-repo.v1",
        token_usage="onex.snapshot.injected.token-usage.v1",
    )


@pytest.mark.unit
def test_from_contract_reads_declared_topics() -> None:
    declared = yaml.safe_load(_CONTRACT.read_text())["config"]["cost_snapshot_topics"]
    topics = ModelCostSnapshotTopics.from_contract()
    assert topics.model_dump() == declared
    assert len(set(topics.model_dump().values())) == 3


@pytest.mark.unit
def test_from_contract_fails_loudly_without_the_section(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text("config: {}\n")
    with pytest.raises(ValidationError):
        ModelCostSnapshotTopics.from_contract(contract)


@pytest.mark.unit
def test_create_app_binds_contract_topics() -> None:
    app = create_app(container=ModelONEXContainer(), cors_origins=["http://testserver"])
    assert app.state.cost_snapshot_topics == ModelCostSnapshotTopics.from_contract()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_summary_route_reads_the_injected_topic() -> None:
    injected = _injected_topics()
    app = create_app(
        container=ModelONEXContainer(),
        cors_origins=["http://testserver"],
        cost_api_pool=_UnusedPool(),
    )
    app.dependency_overrides[get_cost_snapshot_topics] = lambda: injected
    store_latest_snapshot(
        app.state.cost_snapshot_topics.summary,
        "24h",
        {"total_cost_usd": "9.000000", "total_tokens": 99, "session_count": 9},
    )
    store_latest_snapshot(
        injected.summary,
        "24h",
        {"total_cost_usd": "1.000000", "total_tokens": 11, "session_count": 1},
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/api/costs/summary")
        app.dependency_overrides.clear()
        contract_response = await client.get("/api/costs/summary")

    assert response.status_code == 200
    assert response.json()["total_tokens"] == 11
    assert contract_response.status_code == 200
    assert contract_response.json()["total_tokens"] == 99


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handlers_read_snapshots_under_injected_topics() -> None:
    topics = _injected_topics()
    store_latest_snapshot(
        topics.summary,
        "24h",
        {"total_cost_usd": "2.000000", "total_tokens": 20, "session_count": 2},
    )
    store_latest_snapshot(
        topics.by_repo,
        "24h",
        {"rows": [{"repo_name": "r", "cost_usd": "1.0", "total_tokens": 5}]},
    )
    store_latest_snapshot(
        topics.token_usage,
        "24h",
        {"rows": [{"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}]},
    )
    pool = _UnusedPool()

    summary = await cost_handlers.fetch_cost_summary(
        pool,
        window="24h",
        snapshot_topics=topics,
    )
    by_repo = await cost_handlers.fetch_cost_by_repo(
        pool,
        window="24h",
        snapshot_topics=topics,
    )
    usage = await cost_handlers.fetch_token_usage(
        pool,
        window="24h",
        snapshot_topics=topics,
    )

    assert summary.total_tokens == 20
    assert [item.name for item in by_repo.items] == ["r"]
    assert usage.total_tokens == 7


@pytest.mark.unit
@pytest.mark.parametrize(
    "body", ["", "config:\n", "config:\n  cost_snapshot_topics:\n", "- x\n"]
)
def test_from_contract_null_or_malformed_sections_raise_validation_error(
    tmp_path: Path, body: str
) -> None:
    """A null or non-mapping section is a ValidationError, never AttributeError."""
    contract = tmp_path / "contract.yaml"
    contract.write_text(body, encoding="utf-8")
    with pytest.raises(ValidationError):
        ModelCostSnapshotTopics.from_contract(contract)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["summary", "by_repo", "token_usage"])
@pytest.mark.parametrize(
    "invalid_topic",
    [
        "onex.evt.platform.cost-summary.v1",
        "snapshot-topic",
        "onex.snapshot.platform.cost-summary",
        "onex.snapshot.platform.cost-summary.v1.trailing",
    ],
)
def test_from_contract_rejects_invalid_snapshot_topics(
    tmp_path: Path, field: str, invalid_topic: str
) -> None:
    declared = yaml.safe_load(_CONTRACT.read_text())
    declared["config"]["cost_snapshot_topics"][field] = invalid_topic
    contract = tmp_path / "contract.yaml"
    contract.write_text(yaml.safe_dump(declared), encoding="utf-8")
    with pytest.raises(ValidationError, match=field):
        ModelCostSnapshotTopics.from_contract(contract)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "correlation_id", ["invalid", "219391ad-2322-4564-acb1-114e13edeb17"]
)
async def test_unbound_snapshot_topics_return_structured_503(
    correlation_id: str,
) -> None:
    app = FastAPI()
    app.state.cost_api_pool = _UnusedPool()
    app.include_router(router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get(
            "/api/costs/summary", headers={"X-Correlation-ID": correlation_id}
        )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["message"] == "Cost API snapshot topics not configured"
    assert detail["context"]["operation"] == "cost_api_snapshot_topics_lookup"
    if correlation_id != "invalid":
        assert detail["context"]["correlation_id"] == correlation_id
