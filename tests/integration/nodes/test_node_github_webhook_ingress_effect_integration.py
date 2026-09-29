# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Prove webhook contract discovery, resolver wiring, verification and folding.

Uses real contracts, a rendered resolver config on disk and an environment-backed
SecretResolver. Deliveries stay in process; no GitHub, Kafka or database is needed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest
import yaml

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.nodes.node_github_webhook_ingress_effect.handlers import (
    HandlerGitHubWebhookIngress,
)
from omnibase_infra.nodes.node_github_webhook_ingress_effect.handlers import (
    handler_github_webhook_ingress as ingress_module,
)
from omnibase_infra.nodes.node_github_webhook_ingress_effect.models import (
    ModelGitHubPrMergedObservation,
    ModelGitHubPrStateObservation,
    ModelGitHubWebhookDelivery,
)
from omnibase_infra.nodes.node_github_webhook_ingress_effect.webhook_fold import (
    fold_delivery,
)
from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    filter_manifest_for_runtime_profile,
)
from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
    RuntimeProfileName,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver
from omnibase_infra.runtime.service_kernel import (
    _build_runtime_handler_dependencies,
    _github_webhook_ingress_dependencies,
)

pytestmark = pytest.mark.integration

_TEST_SECRET = "integration-webhook-signing-key"


def _resolver_config(tmp_path: Path, *, profile: RuntimeProfileName) -> Path:
    contract_path = (
        Path(__file__).resolve().parents[3]
        / "contracts/services/runtime_policy.contract.yaml"
    )
    contract = ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    )
    path = tmp_path / "secret-resolver.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "enable_convention_fallback": False,
                "mappings": [
                    mapping.model_dump(mode="json")
                    for mapping in contract.profiles[profile].secret_resolver_mappings
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> HandlerGitHubWebhookIngress:
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", _TEST_SECRET)
    dependencies = _build_runtime_handler_dependencies(
        postgres_pool=None,
        kafka_bootstrap_servers=None,
        gateway_secret_resolver_config_path=_resolver_config(tmp_path, profile="dev"),
    )
    assert dependencies is not None
    resolver = dependencies["HandlerGitHubWebhookIngress"]["secret_resolver"]
    assert isinstance(resolver, SecretResolver)
    return HandlerGitHubWebhookIngress(secret_resolver=resolver)


@pytest.fixture
def delivery() -> ModelGitHubWebhookDelivery:
    body = json.dumps(
        {
            "action": "closed",
            "repository": {"full_name": "OmniNode-ai/omnibase_infra"},
            "pull_request": {
                "number": 4227,
                "state": "closed",
                "merged": True,
                "draft": False,
                "title": "feat(OMN-14375): GitHub webhook ingress",
                "closed_at": "2026-09-28T12:00:00Z",
                "merged_at": "2026-09-28T12:00:00Z",
                "merge_commit_sha": "c" * 40,
                "head": {
                    "ref": "jonah/omn-14375-github-webhook-ingress",
                    "sha": "a" * 40,
                },
                "base": {"ref": "dev"},
            },
        }
    ).encode()
    signature = hmac.new(_TEST_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return ModelGitHubWebhookDelivery.model_validate(
        {
            "event": "pull_request",
            "delivery_id": "72d3162e-cc78-11e3-81ab-4c9367dc0958",
            "signature_256": f"sha256={signature}",
            "body_b64": base64.b64encode(body).decode(),
            "received_at": "2026-09-28T12:00:01Z",
        }
    )


def test_contract_is_discovered_for_effects_runtime() -> None:
    contract_path = (
        Path(__file__).resolve().parents[3]
        / "src/omnibase_infra/nodes/node_github_webhook_ingress_effect/contract.yaml"
    )
    manifest = discover_contracts_from_paths([contract_path])
    assert manifest.total_errors == 0
    assert manifest.total_discovered == 1
    effects = filter_manifest_for_runtime_profile(manifest, "effects")
    assert [contract.name for contract in effects.manifest.contracts] == [
        "node_github_webhook_ingress_effect"
    ]
    assert (
        filter_manifest_for_runtime_profile(manifest, "main").manifest.contracts == ()
    )


@pytest.mark.asyncio
async def test_wired_handler_verifies_and_folds_merge(
    handler: HandlerGitHubWebhookIngress, delivery: ModelGitHubWebhookDelivery
) -> None:
    result = await handler.handle(delivery)
    assert len(result.events) == 2
    state, merged = result.events
    assert isinstance(state, ModelGitHubPrStateObservation)
    assert isinstance(merged, ModelGitHubPrMergedObservation)
    assert state.topic == "onex.evt.github.pr-status.v1"
    assert state.entity_id == "OmniNode-ai/omnibase_infra#4227"
    assert state.repo == "OmniNode-ai/omnibase_infra"
    assert state.pr_number == 4227
    assert state.delivery_id == delivery.delivery_id
    assert state.source == "webhook"
    assert state.github_event == "pull_request"
    assert state.as_of == datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
    assert state.triage_state == "merged"
    assert state.merge_queue_state == "MERGED"
    assert state.head_sha == "a" * 40
    assert state.head_ref == "jonah/omn-14375-github-webhook-ingress"
    assert state.base_ref == "dev"
    assert state.title == "feat(OMN-14375): GitHub webhook ingress"
    assert state.is_draft is False
    assert state.ci_status is None
    assert state.review_decision is None
    assert merged.topic == "onex.evt.github.pr-merged.v1"
    assert merged.entity_id == state.entity_id
    assert merged.repo == state.repo
    assert merged.pr_number == state.pr_number
    assert merged.branch == state.head_ref
    assert merged.base_ref == "dev"
    assert merged.ticket == "OMN-14375"
    assert merged.merge_sha == "c" * 40
    assert merged.merged_at == "2026-09-28T12:00:00Z"
    assert isinstance(merged.event_id, UUID)
    assert merged.event_id.version == 5

    replay = await handler.handle(delivery)
    assert replay.events[0] == state
    replay_merged = replay.events[1]
    assert isinstance(replay_merged, ModelGitHubPrMergedObservation)
    assert replay_merged.event_id == merged.event_id


@pytest.mark.asyncio
async def test_wired_handler_refuses_forged_delivery_before_folding(
    handler: HandlerGitHubWebhookIngress, delivery: ModelGitHubWebhookDelivery
) -> None:
    forged = delivery.model_copy(update={"signature_256": "sha256=" + "0" * 64})
    with patch.object(ingress_module, "fold_delivery", wraps=fold_delivery) as fold:
        with pytest.raises(RuntimeHostError, match="signature does not verify"):
            await handler.handle(forged)
        fold.assert_not_called()


@pytest.mark.asyncio
async def test_unmapped_lane_returns_none_and_refuses_before_folding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    delivery: ModelGitHubWebhookDelivery,
) -> None:
    # Even an ambient secret must not open a lane with no explicit mapping.
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", _TEST_SECRET)
    config_path = _resolver_config(tmp_path, profile="stability-test")
    with caplog.at_level(logging.INFO, logger="omnibase_infra.runtime.service_kernel"):
        dependencies = _github_webhook_ingress_dependencies(config_path)
    assert dependencies is None
    record = next(
        r for r in caplog.records if "no secret-resolver mapping" in r.message
    )
    assert record.msg == (
        "GitHub webhook ingress: no secret-resolver mapping found for the "
        "configured webhook secret reference on this lane; the ingress "
        "handler will refuse every delivery"
    )
    assert record.args == ()
    runtime_dependencies = _build_runtime_handler_dependencies(
        postgres_pool=None,
        kafka_bootstrap_servers=None,
        gateway_secret_resolver_config_path=config_path,
    )
    assert (
        runtime_dependencies is None
        or "HandlerGitHubWebhookIngress" not in runtime_dependencies
    )

    handler = HandlerGitHubWebhookIngress()
    with patch.object(ingress_module, "fold_delivery", wraps=fold_delivery) as fold:
        with pytest.raises(RuntimeHostError, match="no webhook secret configured"):
            await handler.handle(delivery)
        fold.assert_not_called()
