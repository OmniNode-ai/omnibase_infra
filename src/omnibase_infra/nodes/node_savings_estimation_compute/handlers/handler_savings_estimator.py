# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for computing tiered token savings attribution.

Computes how much token spend the ONEX platform saved in a session by
comparing actual costs against a counterfactual reference model.

Savings are broken into 5 categories across two tiers:

Tier A (direct, high confidence):
    1. Local routing -- calls routed to $0 local models instead of paid APIs.

Tier B (heuristic, varying confidence):
    2. Pattern injection -- learned patterns injected into prompts.
    3. Agent delegation -- avoided unnecessary subagent calls (Phase A: 0).
    4. Memory/RAG -- retrieval avoiding regeneration (Phase A: 0).
    5. Validator catches -- errors caught before reaching LLM regen cycles.

Related Tickets:
    - OMN-5547: Create HandlerSavingsEstimator compute handler
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.models.pricing.model_pricing_table import ModelPricingTable
from omnibase_infra.nodes.node_savings_estimation_compute.models import (
    ModelSavingsInput,
    ModelValidatorCatchSignal,
)

logger = logging.getLogger(__name__)

# Confidence values for validator catch categories
_CATCH_CONFIDENCE: dict[str, float] = {
    "error_poly_enforcer": 0.9,
    "error_pre_commit": 0.7,
    "error_ci": 0.7,
    "error_code_review": 0.7,
    "warning_pre_commit": 0.5,
    "warning_code_review": 0.5,
}
_DEFAULT_CATCH_CONFIDENCE: float = 0.5


class SavingsCategoryResult:
    """Internal accumulator for a single savings category."""

    __slots__ = (
        "category",
        "confidence",
        "cost_saved_usd",
        "evidence",
        "method",
        "tier",
        "tokens_saved",
    )

    def __init__(
        self,
        category: str,
        tier: str,
        tokens_saved: int,
        cost_saved_usd: float,
        confidence: float,
        method: str,
        # ONEX_EXCLUDE: any_type - dict[str, ...] for flexible evidence payload
        evidence: dict[str, object],
    ) -> None:
        self.category = category
        self.tier = tier
        self.tokens_saved = tokens_saved
        self.cost_saved_usd = cost_saved_usd
        self.confidence = confidence
        self.method = method
        self.evidence = evidence


class HandlerSavingsEstimator:
    """Compute tiered token savings attribution for a session.

    Stateless handler. Receives session signals and a pricing table,
    computes savings across 5 categories, and returns a structured
    estimate matching the ContractSavingsEstimate wire format.
    """

    def __init__(self, pricing_table: ModelPricingTable) -> None:
        self._pricing_table = pricing_table

    @property
    def handler_id(self) -> str:
        return "handler-savings-estimator"

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE

    # ONEX_EXCLUDE: any_type - dict[str, Any] return for SPI contract wire format
    async def handle(self, savings_input: ModelSavingsInput) -> dict[str, object]:
        """Compute savings estimate from session signals.

        Returns a dict matching the ContractSavingsEstimate wire format.
        """
        config = savings_input.baseline_config
        categories: list[SavingsCategoryResult] = []
        has_unknown_model = False

        # --- Category 1: Local routing (Tier A) ---
        cat1 = self._compute_local_routing(savings_input)
        if cat1.tokens_saved > 0 or savings_input.llm_calls:
            categories.append(cat1)
        if any(
            not self._pricing_table.has_model(c.model_id)
            for c in savings_input.llm_calls
        ):
            has_unknown_model = True

        # --- Category 2: Pattern injection (Tier B) ---
        cat2 = self._compute_pattern_injection(savings_input)
        if cat2.tokens_saved > 0:
            categories.append(cat2)

        # --- Category 3: Agent delegation (Tier B, Phase A: 0) ---
        cat3 = self._compute_delegation(savings_input)
        categories.append(cat3)

        # --- Category 4: Memory/RAG (Tier B, Phase A: 0) ---
        cat4 = self._compute_rag(savings_input)
        categories.append(cat4)

        # --- Category 5: Validator catches (Tier B) ---
        cat5 = self._compute_validator_catches(savings_input)
        if cat5.tokens_saved > 0:
            categories.append(cat5)

        # Aggregate
        direct_cats = [c for c in categories if c.tier == "direct"]
        heuristic_cats = [
            c for c in categories if c.tier == "heuristic" and c.confidence > 0.0
        ]

        direct_savings_usd = sum(c.cost_saved_usd for c in direct_cats)
        direct_tokens_saved = sum(c.tokens_saved for c in direct_cats)
        total_savings_usd = sum(c.cost_saved_usd for c in categories)
        total_tokens_saved = sum(c.tokens_saved for c in categories)

        direct_confidence = direct_cats[0].confidence if direct_cats else 0.0
        if heuristic_cats:
            total_heuristic_cost = sum(c.cost_saved_usd for c in heuristic_cats)
            if total_heuristic_cost > 0:
                heuristic_confidence_avg = (
                    sum(c.confidence * c.cost_saved_usd for c in heuristic_cats)
                    / total_heuristic_cost
                )
            else:
                heuristic_confidence_avg = sum(
                    c.confidence for c in heuristic_cats
                ) / len(heuristic_cats)
        else:
            heuristic_confidence_avg = 0.0

        # Actual session totals
        actual_total_tokens = sum(
            c.prompt_tokens + c.completion_tokens for c in savings_input.llm_calls
        )
        actual_cost_usd = 0.0
        for call in savings_input.llm_calls:
            est = self._pricing_table.estimate_cost(
                call.model_id, call.prompt_tokens, call.completion_tokens
            )
            if est.estimated_cost_usd is not None:
                actual_cost_usd += est.estimated_cost_usd

        completeness_status = "complete"
        if has_unknown_model:
            completeness_status = "partial"
        elif not savings_input.delegation_signals and not savings_input.rag_signals:
            completeness_status = "phase_limited"

        actual_model_ids = list({c.model_id for c in savings_input.llm_calls})

        logger.info(
            "Savings estimate: session=%s direct=%.4f heuristic=%.4f total=%.4f",
            savings_input.session_id,
            direct_savings_usd,
            total_savings_usd - direct_savings_usd,
            total_savings_usd,
        )

        return {
            "schema_version": "1.0",
            "session_id": savings_input.session_id,
            "correlation_id": savings_input.correlation_id,
            "timestamp_iso": datetime.now(tz=UTC).isoformat(),
            "actual_total_tokens": actual_total_tokens,
            "actual_cost_usd": round(actual_cost_usd, 10),
            "actual_model_id": actual_model_ids[0] if actual_model_ids else "",
            "counterfactual_model_id": config.reference_model_id,
            "direct_savings_usd": round(direct_savings_usd, 10),
            "direct_tokens_saved": direct_tokens_saved,
            "estimated_total_savings_usd": round(total_savings_usd, 10),
            "estimated_total_tokens_saved": total_tokens_saved,
            "categories": [
                {
                    "category": c.category,
                    "tier": c.tier,
                    "tokens_saved": c.tokens_saved,
                    "cost_saved_usd": round(c.cost_saved_usd, 10),
                    "confidence": c.confidence,
                    "method": c.method,
                    "evidence": c.evidence,
                }
                for c in categories
            ],
            "direct_confidence": direct_confidence,
            "heuristic_confidence_avg": round(heuristic_confidence_avg, 4),
            "estimation_method": "tiered_attribution_v1",
            "treatment_group": savings_input.treatment_group,
            "is_measured": False,
            "measurement_basis": "",
            "baseline_session_id": "",
            "pricing_manifest_version": config.pricing_manifest_version,
            "completeness_status": completeness_status,
            "extensions": {},
        }

    # ------------------------------------------------------------------
    # Category computation methods
    # ------------------------------------------------------------------

    def _compute_local_routing(
        self, savings_input: ModelSavingsInput
    ) -> SavingsCategoryResult:
        """Category 1: Local routing savings (Tier A, confidence 1.0)."""
        config = savings_input.baseline_config
        total_tokens_saved = 0
        total_cost_saved = 0.0
        call_count = 0

        for call in savings_input.llm_calls:
            entry = self._pricing_table.get_entry(call.model_id)
            if entry is None:
                continue
            if entry.input_cost_per_1k == 0.0 and entry.output_cost_per_1k == 0.0:
                # Local model -- compute what reference model would have cost
                ref_estimate = self._pricing_table.estimate_cost(
                    config.reference_model_id,
                    call.prompt_tokens,
                    call.completion_tokens,
                )
                if ref_estimate.estimated_cost_usd is not None:
                    total_cost_saved += ref_estimate.estimated_cost_usd
                    total_tokens_saved += call.prompt_tokens + call.completion_tokens
                    call_count += 1

        return SavingsCategoryResult(
            category="local_routing",
            tier="direct",
            tokens_saved=total_tokens_saved,
            cost_saved_usd=total_cost_saved,
            confidence=1.0 if call_count > 0 else 0.0,
            method="reference_pricing_delta",
            evidence={
                "evidence_type": "local_routing",
                "reference_model_id": config.reference_model_id,
                "reference_cost_usd": round(total_cost_saved, 10),
                "actual_model_id": "local_mix",
                "actual_cost_usd": 0.0,
                "call_count": call_count,
            },
        )

    def _compute_pattern_injection(
        self, savings_input: ModelSavingsInput
    ) -> SavingsCategoryResult:
        """Category 2: Pattern injection savings (Tier B, confidence 0.8)."""
        config = savings_input.baseline_config
        total_tokens_saved = 0
        total_cost_saved = 0.0
        patterns_injected = 0
        tokens_injected = 0

        for signal in savings_input.injection_signals:
            saved = int(signal.tokens_injected * config.regen_multiplier)
            total_tokens_saved += saved
            patterns_injected += signal.patterns_count
            tokens_injected += signal.tokens_injected

            ref_estimate = self._pricing_table.estimate_cost(
                config.reference_model_id, saved, 0
            )
            if ref_estimate.estimated_cost_usd is not None:
                total_cost_saved += ref_estimate.estimated_cost_usd

        return SavingsCategoryResult(
            category="pattern_injection",
            tier="heuristic",
            tokens_saved=total_tokens_saved,
            cost_saved_usd=total_cost_saved,
            confidence=0.8 if savings_input.injection_signals else 0.0,
            method="regen_multiplier",
            evidence={
                "evidence_type": "pattern_injection",
                "patterns_injected": patterns_injected,
                "tokens_injected": tokens_injected,
                "regen_multiplier": config.regen_multiplier,
            },
        )

    def _compute_delegation(
        self, savings_input: ModelSavingsInput
    ) -> SavingsCategoryResult:
        """Category 3: Agent delegation savings (Tier B, Phase A: 0)."""
        return SavingsCategoryResult(
            category="agent_delegation",
            tier="heuristic",
            tokens_saved=0,
            cost_saved_usd=0.0,
            confidence=0.0,
            method="delegation_avoidance",
            evidence={
                "evidence_type": "agent_delegation",
                "subagent_calls_avoided": 0,
                "avg_tokens_per_call": savings_input.baseline_config.avg_tokens_per_subagent_call,
                "baseline_version": savings_input.baseline_config.baseline_version,
            },
        )

    def _compute_rag(self, savings_input: ModelSavingsInput) -> SavingsCategoryResult:
        """Category 4: Memory/RAG savings (Tier B, Phase A: 0)."""
        return SavingsCategoryResult(
            category="memory_rag",
            tier="heuristic",
            tokens_saved=0,
            cost_saved_usd=0.0,
            confidence=0.0,
            method="rag_regen_avoidance",
            evidence={
                "evidence_type": "memory_rag",
                "tokens_retrieved": 0,
                "regen_tokens_estimate": 0,
                "regen_multiplier": savings_input.baseline_config.rag_regen_multiplier,
                "baseline_version": savings_input.baseline_config.baseline_version,
            },
        )

    def _compute_validator_catches(
        self, savings_input: ModelSavingsInput
    ) -> SavingsCategoryResult:
        """Category 5: Validator catch savings (Tier B, severity-weighted)."""
        config = savings_input.baseline_config
        total_tokens_saved = 0
        total_cost_saved = 0.0
        catches_by_severity: dict[str, int] = {}
        catches_by_type: dict[str, int] = {}
        weighted_confidence_sum = 0.0
        catch_count = 0

        for catch in savings_input.validator_catches:
            key = f"{catch.severity}_{catch.validator_type}"
            tokens = config.tokens_per_fix_cycle_by_severity.get(key, 0)
            total_tokens_saved += tokens
            catch_count += 1

            catches_by_severity[catch.severity] = (
                catches_by_severity.get(catch.severity, 0) + 1
            )
            catches_by_type[catch.validator_type] = (
                catches_by_type.get(catch.validator_type, 0) + 1
            )

            confidence = _CATCH_CONFIDENCE.get(key, _DEFAULT_CATCH_CONFIDENCE)
            weighted_confidence_sum += confidence * tokens

            ref_estimate = self._pricing_table.estimate_cost(
                config.reference_model_id, tokens, 0
            )
            if ref_estimate.estimated_cost_usd is not None:
                total_cost_saved += ref_estimate.estimated_cost_usd

        avg_confidence = (
            weighted_confidence_sum / total_tokens_saved
            if total_tokens_saved > 0
            else 0.0
        )

        return SavingsCategoryResult(
            category="validator_catches",
            tier="heuristic",
            tokens_saved=total_tokens_saved,
            cost_saved_usd=total_cost_saved,
            confidence=round(avg_confidence, 4) if catch_count > 0 else 0.0,
            method="severity_weighted_fix_cycles",
            evidence={
                "evidence_type": "validator_catches",
                "catch_count": catch_count,
                "catches_by_severity": catches_by_severity,
                "catches_by_type": catches_by_type,
                "tokens_per_fix_cycle_weighted": (
                    total_tokens_saved // catch_count if catch_count > 0 else 0
                ),
                "fix_cycle_baseline_version": config.baseline_version,
            },
        )

    @staticmethod
    def _catch_confidence(catch: ModelValidatorCatchSignal) -> float:
        """Get confidence for a validator catch based on severity and type."""
        key = f"{catch.severity}_{catch.validator_type}"
        return _CATCH_CONFIDENCE.get(key, _DEFAULT_CATCH_CONFIDENCE)


__all__: list[str] = ["HandlerSavingsEstimator"]
