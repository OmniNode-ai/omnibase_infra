# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the /health(/detailed) <- boot attach-readiness join (OMN-15642).

OMN-15642 observed a live regression on onex-dev (deploy-onex-staging runs
30720296789 and, on n=2 reproduction, 30721670043): the corrected tree booted
cleanly -- steps 30-36 (rollout, digest triple-match, dashboard health,
post-deploy verification, staleness) all passed -- while a correlation-keyed
delegation projection and the unified system-event stream stayed silently
empty, surfacing only ~60s later at the terminal business-proof gate
(OMN-15256). ``subscribe_wired_contract_topics`` already classifies exactly
this shape of failure (a non-core consumer contract whose Kafka subscription
attach failed or was skipped) as ``EnumRuntimeReadinessState.DEGRADED`` via
``ModelRuntimeAttachReadiness`` -- OMN-15512 wired that aggregate onto
``ServiceHealth`` -- but pre-fix it reached only
``details.components.runtime_wiring``, a nested payload detail nothing
upstream reads. The top-level ``status`` field stayed "healthy" no matter how
degraded the boot attach was.

REMEDIATION (same ticket, second round): an adversarial verifier found that
folding this aggregate into ``/health``'s gated ``status`` -- the first
version of this fix -- is a live deploy-wedge risk, not a safe visibility fix.
``ModelRuntimeAttachReadiness``'s own docstring says "the readiness endpoint
reports attach status ONLY -- it is not a source of truth for contract
lifecycle", and OMN-13237 deliberately made a NOT_READY/DEGRADED contract
non-fatal (recorded and skipped, the process is never recycled for it). But
four real automated consumers hard-gate boot/deploy readiness on
``/health``'s ``status`` with no DEGRADED tolerance --
``.github/workflows/reusable-runtime-boot.yml``,
``scripts/deploy-agent/deploy_agent/executor.py``, and
``scripts/runtime_build/verify_stability_refresh.py`` /
``verify_dev_refresh.py``. A single unprovisioned topic (a documented live
condition on onex-dev, OMN-15330) would have turned every ordinary NOT_READY
skip into a hard boot/deploy failure. The fold now lands ONLY on
``/health/detailed`` (verified: no k8s manifest, CI workflow, or deploy script
in this repo or ``omninode_infra`` reads that path), which still closes the
original visibility gap for a human or dashboard checking after steps 30-36 --
just not for the four automated hard gates. ``/health`` itself is pinned back
to its pre-OMN-15642 behavior: attach readiness never changes its ``status``.

This module still mirrors ``test_service_health_runtime_verdict.py``'s
``/health``-side shape for the "no aggregate yet" and "fully attached" cases,
which ARE endpoint-symmetric; only the DEGRADED/FAILED cases diverge per the
scoping decision above.

Related Tickets:
    - OMN-15642: onex-dev correlation-keyed projection/system-event regression
    - OMN-15512: boot attach-readiness aggregate riding the /health payload
    - OMN-15217: precedent -- runtime-health-monitor verdict join
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.event_bus.enum_runtime_readiness_state import (
    EnumRuntimeReadinessState,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.event_bus.model_runtime_attach_readiness import (
    ModelRuntimeAttachReadiness,
)
from omnibase_infra.services.health_checker import ServiceHealth


def _live_runtime() -> MagicMock:
    """A runtime whose process-local health is perfect.

    This is exactly what RuntimeHostProcess.health_check() reports when a
    Kafka consumer contract silently fails its subscription attach: the
    process itself never sees the per-contract wiring outcome.
    """
    runtime = MagicMock()
    runtime.health_check = AsyncMock(
        return_value={
            "healthy": True,
            "degraded": False,
            "is_running": True,
            "runtime_attached": True,
            "event_bus_healthy": True,
        }
    )
    return runtime


def _readiness(
    *,
    attached: int,
    required: int,
    failing_contract: str = "node_projection_delegation",
) -> ModelRuntimeAttachReadiness:
    """Build an aggregate with ``required - attached`` non-attached contracts.

    Mirrors the live shape: every non-attached result uses FAILED (the
    per-contract outcome ``subscribe_wired_contract_topics`` records when a
    contract's dispatcher-scope guard or Kafka subscribe raises after
    readiness passed -- caught, non-fatal, see
    ``omnibase_infra.runtime.auto_wiring.handler_wiring``'s per-contract
    try/except around ``_subscribe_contract_topics``).
    """
    results = [
        ModelContractAttachResult(
            contract_name=f"node_attached_{i}",
            status=EnumContractAttachStatus.ATTACHED,
            topics_subscribed=("onex.evt.omnimarket.attached-topic.v1",),
        )
        for i in range(attached)
    ]
    results.extend(
        ModelContractAttachResult(
            contract_name=(failing_contract if i == 0 else f"{failing_contract}_{i}"),
            status=EnumContractAttachStatus.FAILED,
            detail="handler_wiring: contract-scoped subscription has an empty "
            "or invalid dispatcher scope; refusing process-global fan-out.",
        )
        for i in range(required - attached)
    )
    return ModelRuntimeAttachReadiness.from_results(tuple(results))


async def _get_health(server: ServiceHealth) -> tuple[int, dict]:
    response = await server._handle_health(MagicMock(spec=web.Request))
    assert response.text is not None
    return response.status, json.loads(response.text)


async def _get_health_detailed(server: ServiceHealth) -> tuple[int, dict]:
    response = await server._handle_health_detailed(MagicMock(spec=web.Request))
    assert response.text is not None
    return response.status, json.loads(response.text)


@pytest.mark.unit
class TestAttachReadinessJoin:
    @pytest.mark.asyncio
    async def test_without_an_aggregate_the_payload_cannot_see_the_gap(self) -> None:
        """The pre-attach state, pinned: no aggregate means no signal yet.

        Mirrors OMN-15217's identical "absent verdict" case for the
        runtime-health-monitor join -- absence is not evidence of
        degradation, and this is the state before the kernel's first
        ``subscribe_wired_contract_topics`` call completes.
        """
        server = ServiceHealth(runtime=_live_runtime(), version="1.0.0")

        status, body = await _get_health(server)

        assert status == 200
        assert body["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_degraded_attach_readiness_does_not_degrade_gated_health_status(
        self,
    ) -> None:
        """REMEDIATION (RED against the first OMN-15642 fix, GREEN after).

        A DEGRADED attach aggregate must NOT flip ``/health``'s gated
        ``status`` -- see the module docstring's "REMEDIATION" section for the
        deploy-wedge this closes. RED against the merged-then-reverted
        version of this fix (commit 363120f4c, dev): that version folded
        ``fold_attach_readiness_into_status`` into ``_handle_health`` too, so
        this assertion read "degraded", not "healthy" (proven by re-adding
        that fold locally and re-running: fails on this exact line). The
        aggregate itself is still fully visible, just not through the gated
        field -- via ``details.degraded`` (a pre-existing, non-gated flag) and
        the nested ``components.runtime_wiring`` detail (OMN-15512).
        """
        server = ServiceHealth(runtime=_live_runtime(), version="1.0.0")
        server.attach_readiness(_readiness(attached=40, required=41))

        status, body = await _get_health(server)

        assert body["status"] == "healthy"
        assert status == 200
        # The aggregate did attach and IS visible -- just not gated.
        assert body["details"]["components"]["runtime_wiring"]["status"] == "degraded"

        # /health/detailed, unlike /health, is not a boot/deploy probe target
        # for any of the four hard-gate consumers -- the fold stays wired
        # there, so the SAME aggregate is honestly visible in its status.
        detailed_status, detailed_body = await _get_health_detailed(server)
        assert detailed_body["status"] == "degraded"
        assert detailed_status == 200

    @pytest.mark.asyncio
    async def test_ready_attach_aggregate_leaves_a_healthy_runtime_healthy(
        self,
    ) -> None:
        server = ServiceHealth(runtime=_live_runtime(), version="1.0.0")
        server.attach_readiness(_readiness(attached=41, required=41))

        status, body = await _get_health(server)

        assert status == 200
        assert body["status"] == "healthy"
        assert body["details"]["components"]["runtime_wiring"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_core_gap_does_not_flip_gated_health_status_either(
        self,
    ) -> None:
        """A FAILED aggregate (core control-plane contract gap) ALSO does not
        touch /health's gated status -- the fold is fully unwired from
        ``_handle_health`` for both readiness states, not just DEGRADED, so
        there is no partial/inconsistent behavior depending on severity.
        ``/health/detailed`` still carries the fold (pinned symmetrically
        below and in ``test_detailed_endpoint_flips_http_status_on_core_gap``).
        """
        readiness = ModelRuntimeAttachReadiness(
            state=EnumRuntimeReadinessState.FAILED,
            required_contracts=5,
            attached_contracts=4,
            results=(
                ModelContractAttachResult(
                    contract_name="node_core_control_plane",
                    status=EnumContractAttachStatus.FAILED,
                ),
            ),
        )
        server = ServiceHealth(runtime=_live_runtime(), version="1.0.0")
        server.attach_readiness(readiness)

        status, body = await _get_health(server)

        assert body["status"] == "healthy"
        assert status == 200

    @pytest.mark.asyncio
    async def test_detailed_endpoint_also_folds_attach_readiness(self) -> None:
        """/health/detailed's own status/http_status pair carries the same fix.

        Unlike /health, /health/detailed is not a k8s probe target (verified:
        no k8s manifest under k8s/onex-dev references it), so it is safe and
        consistent with this endpoint's own documented contract ("503:
        Unhealthy") to flip http_status too.
        """
        server = ServiceHealth(runtime=_live_runtime(), version="1.0.0")
        server.attach_readiness(_readiness(attached=40, required=41))

        status, body = await _get_health_detailed(server)

        assert body["status"] == "degraded"
        assert status == 200

    @pytest.mark.asyncio
    async def test_detailed_endpoint_flips_http_status_on_core_gap(self) -> None:
        readiness = ModelRuntimeAttachReadiness(
            state=EnumRuntimeReadinessState.FAILED,
            required_contracts=5,
            attached_contracts=4,
            results=(
                ModelContractAttachResult(
                    contract_name="node_core_control_plane",
                    status=EnumContractAttachStatus.FAILED,
                ),
            ),
        )
        server = ServiceHealth(runtime=_live_runtime(), version="1.0.0")
        server.attach_readiness(readiness)

        status, body = await _get_health_detailed(server)

        assert body["status"] == "unhealthy"
        assert status == 503
