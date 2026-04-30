# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compute runner cost avoidance from runner usage events."""

from __future__ import annotations

from collections import OrderedDict
from decimal import ROUND_HALF_UP, Decimal

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.pricing import ModelPricingTable
from omnibase_infra.nodes.node_runner_usage_effect.models import (
    ModelRunnerSavingsEstimated,
    ModelRunnerUsageEvent,
)

_USD_QUANT = Decimal("0.000001")
_MINUTES_QUANT = Decimal("0.000001")
_DEFAULT_EMITTED_KEY_LIMIT = 10_000


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(_USD_QUANT, rounding=ROUND_HALF_UP)


class HandlerRunnerUsageSavings:
    """Estimate avoided GitHub-hosted runner cost for self-hosted jobs."""

    def __init__(
        self,
        pricing_table: ModelPricingTable | None = None,
        *,
        emitted_key_limit: int = _DEFAULT_EMITTED_KEY_LIMIT,
    ) -> None:
        if emitted_key_limit < 1:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="configure_runner_usage_savings",
                target_name="handler-runner-usage-savings",
            )
            raise ProtocolConfigurationError(
                "emitted_key_limit must be greater than zero",
                context=context,
                parameter="emitted_key_limit",
                value=emitted_key_limit,
            )
        self._pricing_table = pricing_table or ModelPricingTable.from_yaml()
        self._emitted_key_limit = emitted_key_limit
        self._emitted_keys: OrderedDict[tuple[str, str], None] = OrderedDict()

    @property
    def handler_id(self) -> str:
        return "handler-runner-usage-savings"

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        usage_event: ModelRunnerUsageEvent,
    ) -> ModelRunnerSavingsEstimated | None:
        """Return one savings estimate for each new runner usage idempotency key."""
        return self.compute_savings(usage_event)

    def compute_savings(
        self,
        usage_event: ModelRunnerUsageEvent,
    ) -> ModelRunnerSavingsEstimated | None:
        """Compute a runner savings event, deduped by workflow run and job."""
        key = usage_event.idempotency_key
        if key in self._emitted_keys:
            self._emitted_keys.move_to_end(key)
            return None

        runner_cost = self._pricing_table.runner_cost
        if runner_cost is None:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="compute_runner_usage_savings",
                target_name=self.handler_id,
            )
            raise ProtocolConfigurationError(
                "Pricing manifest missing runner_cost.github_hosted_per_minute_usd",
                context=context,
                parameter="runner_cost.github_hosted_per_minute_usd",
            )

        runner_minutes = Decimal(str(usage_event.runner_minutes)).quantize(
            _MINUTES_QUANT,
            rounding=ROUND_HALF_UP,
        )
        hosted_rate = Decimal(str(runner_cost.github_hosted_per_minute_usd))
        cloud_cost = _quantize(runner_minutes * hosted_rate)
        local_cost = Decimal("0.000000")

        estimate = ModelRunnerSavingsEstimated(
            source_event_id=f"runner-usage:{key[0]}:{key[1]}",
            event_timestamp=usage_event.event_timestamp,
            session_id=usage_event.session_id,
            workflow_run_id=usage_event.workflow_run_id,
            job_id=usage_event.job_id,
            runner_minutes=runner_minutes,
            local_cost_usd=local_cost,
            cloud_cost_usd=cloud_cost,
            savings_usd=cloud_cost - local_cost,
            repo_name=usage_event.repo_name,
            machine_id=usage_event.machine_id,
            runner_name=usage_event.runner_name,
            workflow_name=usage_event.workflow_name,
            pricing_manifest_version=self._pricing_table.schema_version,
        )
        self._remember_emitted_key(key)
        return estimate

    def _remember_emitted_key(self, key: tuple[str, str]) -> None:
        """Track emitted keys with bounded memory use."""
        self._emitted_keys[key] = None
        self._emitted_keys.move_to_end(key)
        while len(self._emitted_keys) > self._emitted_key_limit:
            self._emitted_keys.popitem(last=False)

    def replay(
        self,
        usage_events: list[ModelRunnerUsageEvent],
    ) -> list[ModelRunnerSavingsEstimated]:
        """Replay a fixture stream through the same idempotent handler."""
        estimates: list[ModelRunnerSavingsEstimated] = []
        for event in usage_events:
            estimate = self.compute_savings(event)
            if estimate is not None:
                estimates.append(estimate)
        return estimates


__all__: list[str] = ["HandlerRunnerUsageSavings"]
