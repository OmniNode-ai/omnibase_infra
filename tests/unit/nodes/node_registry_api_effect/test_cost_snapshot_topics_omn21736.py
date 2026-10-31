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
from pydantic import ValidationError

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.nodes.node_registry_api_effect import load_registry_api_config
from omnibase_infra.nodes.node_registry_api_effect.models import (
    ModelCostSnapshotTopics,
)
from omnibase_infra.services.cost_api import handlers as cost_handlers
from omnibase_infra.services.cost_api.router import get_cost_snapshot_topics
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
        summary="injected-summary",
        by_repo="injected-by-repo",
        token_usage="injected-token-usage",
    )


@pytest.mark.unit
def test_from_contract_reads_declared_topics() -> None:
    declared = yaml.safe_load(_CONTRACT.read_text())["config"]["cost_snapshot_topics"]
    topics = ModelCostSnapshotTopics.from_contract()
    assert topics.model_dump() == declared
    assert len(set(topics.model_dump().values())) == 3


@pytest.mark.unit
def test_from_contract_fails_loudly_without_the_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnibase_infra.nodes.node_registry_api_effect.models import (
        model_cost_snapshot_topics as module,
    )

    monkeypatch.setattr(module, "load_registry_api_config", dict)
    with pytest.raises(ValidationError):
        ModelCostSnapshotTopics.from_contract()


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
        injected.summary,
        "24h",
        {"total_cost_usd": "1.000000", "total_tokens": 11, "session_count": 1},
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/api/costs/summary")

    assert response.status_code == 200
    assert response.json()["total_tokens"] == 11


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
