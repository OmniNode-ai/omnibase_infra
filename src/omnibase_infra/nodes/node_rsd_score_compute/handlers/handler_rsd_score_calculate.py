# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that calculates RSD 5-factor priority scores.

This is a COMPUTE handler - pure transformation, no I/O.
Ported from Archive/omnibase_5/rsd-priority-engine CalculatorRSDPriority.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timezone
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_agent_request_data import (
    ModelAgentRequestData,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_dependency_edge import (
    ModelDependencyEdge,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_plan_override_data import (
    ModelPlanOverrideData,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_ticket_data import (
    ModelTicketData,
)
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_factor_score import (
    ModelRsdFactorScore,
)
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_factor_weights import (
    ModelRsdFactorWeights,
)
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_score_result import (
    ModelRsdScoreResult,
)
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_ticket_score import (
    ModelRsdTicketScore,
)

logger = logging.getLogger(__name__)


class HandlerRsdScoreCalculate:
    """Pure RSD 5-factor priority scoring.

    Factors and default weights:
        - 40% dependency distance: how many tickets are blocked downstream
        - 25% failure surface: risk based on ticket keywords and tags
        - 15% time decay: older tickets get higher priority (anti-starvation)
        - 10% agent utility: weighted agent request frequency
        - 10% user weighting: manual priority overrides with time decay
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        correlation_id: UUID,
        tickets: tuple[ModelTicketData, ...],
        dependency_edges: tuple[ModelDependencyEdge, ...],
        agent_requests: tuple[ModelAgentRequestData, ...],
        plan_overrides: tuple[ModelPlanOverrideData, ...],
        weights: ModelRsdFactorWeights,
    ) -> ModelRsdScoreResult:
        """Calculate RSD priority scores for all tickets.

        Args:
            correlation_id: Workflow correlation ID.
            tickets: Ticket data to score.
            dependency_edges: Dependency graph edges.
            agent_requests: Agent requests per ticket.
            plan_overrides: Plan overrides per ticket.
            weights: Factor weights for the algorithm.

        Returns:
            ModelRsdScoreResult with scores and ranked order.
        """
        logger.info(
            "Calculating RSD scores for %d tickets (correlation_id=%s)",
            len(tickets),
            correlation_id,
        )

        # Build lookup structures
        edges_by_source: dict[str, list[ModelDependencyEdge]] = {}
        for edge in dependency_edges:
            edges_by_source.setdefault(edge.source_id, []).append(edge)

        requests_by_ticket: dict[str, list[ModelAgentRequestData]] = {}
        for req in agent_requests:
            requests_by_ticket.setdefault(req.ticket_id, []).append(req)

        overrides_by_ticket: dict[str, list[ModelPlanOverrideData]] = {}
        for ovr in plan_overrides:
            overrides_by_ticket.setdefault(ovr.ticket_id, []).append(ovr)

        all_ticket_ids = {t.ticket_id for t in tickets}

        # Score each ticket
        scored: list[ModelRsdTicketScore] = []
        for ticket in tickets:
            ticket_edges = edges_by_source.get(ticket.ticket_id, [])
            ticket_requests = requests_by_ticket.get(ticket.ticket_id, [])
            ticket_overrides = overrides_by_ticket.get(ticket.ticket_id, [])

            dep_score = _calculate_dependency_distance(
                ticket, ticket_edges, edges_by_source, all_ticket_ids
            )
            fail_score = _calculate_failure_surface(ticket)
            time_score = _calculate_time_decay(ticket)
            agent_score = _calculate_agent_utility(ticket_requests)
            user_score = _calculate_user_weighting(ticket, ticket_overrides)

            factors = (
                ModelRsdFactorScore(
                    factor_name="dependency_distance",
                    raw_score=dep_score,
                    weight=weights.dependency_distance,
                    weighted_score=dep_score * weights.dependency_distance,
                ),
                ModelRsdFactorScore(
                    factor_name="failure_surface",
                    raw_score=fail_score,
                    weight=weights.failure_surface,
                    weighted_score=fail_score * weights.failure_surface,
                ),
                ModelRsdFactorScore(
                    factor_name="time_decay",
                    raw_score=time_score,
                    weight=weights.time_decay,
                    weighted_score=time_score * weights.time_decay,
                ),
                ModelRsdFactorScore(
                    factor_name="agent_utility",
                    raw_score=agent_score,
                    weight=weights.agent_utility,
                    weighted_score=agent_score * weights.agent_utility,
                ),
                ModelRsdFactorScore(
                    factor_name="user_weighting",
                    raw_score=user_score,
                    weight=weights.user_weighting,
                    weighted_score=user_score * weights.user_weighting,
                ),
            )

            final = sum(f.weighted_score for f in factors)
            final = max(0.0, min(1.0, final))

            scored.append(
                ModelRsdTicketScore(
                    ticket_id=ticket.ticket_id,
                    final_score=final,
                    factors=factors,
                )
            )

        # Rank by descending score
        ranked = sorted(scored, key=lambda s: s.final_score, reverse=True)
        ranked_ids = tuple(s.ticket_id for s in ranked)

        logger.info(
            "RSD scoring complete: %d tickets scored, top=%s (%.3f)",
            len(ranked),
            ranked_ids[0] if ranked_ids else "none",
            ranked[0].final_score if ranked else 0.0,
        )

        return ModelRsdScoreResult(
            correlation_id=correlation_id,
            ticket_scores=tuple(ranked),
            ranked_ticket_ids=ranked_ids,
        )


# ---------------------------------------------------------------------------
# Pure scoring functions (no I/O, no state)
# ---------------------------------------------------------------------------


def _count_downstream(
    ticket_id: str,
    edges_by_source: dict[str, list[ModelDependencyEdge]],
    visited: set[str] | None = None,
) -> int:
    """Count all downstream dependent tickets (recursive, cycle-safe)."""
    if visited is None:
        visited = set()
    if ticket_id in visited:
        return 0
    visited.add(ticket_id)
    count = 0
    for edge in edges_by_source.get(ticket_id, []):
        count += 1
        count += _count_downstream(edge.target_id, edges_by_source, visited)
    return count


def _calculate_dependency_distance(
    ticket: ModelTicketData,
    ticket_edges: list[ModelDependencyEdge],
    edges_by_source: dict[str, list[ModelDependencyEdge]],
    all_ticket_ids: set[str],
) -> float:
    """Dependency distance factor: tickets blocking many others score higher.

    Sub-factors (weighted):
        - 30% direct dependents count (capped at 5)
        - 30% total downstream count (capped at 20)
        - 30% whether ticket is a bottleneck (blocks > median)
        - 10% inverse average distance
    """
    direct = len(ticket_edges)
    downstream = _count_downstream(ticket.ticket_id, edges_by_source)

    # Bottleneck heuristic: blocks more than 2 tickets
    is_bottleneck = direct >= 3

    # Average distance is always 1 for direct edges
    avg_distance = 1.0 if direct > 0 else 0.0

    direct_score = min(1.0, direct / 5.0)
    downstream_score = min(1.0, downstream / 20.0)
    bottleneck_score = 1.0 if is_bottleneck else 0.0
    distance_score = 1.0 / (1.0 + avg_distance) if direct > 0 else 0.5

    return (
        direct_score * 0.3
        + downstream_score * 0.3
        + bottleneck_score * 0.3
        + distance_score * 0.1
    )


def _calculate_failure_surface(ticket: ModelTicketData) -> float:
    """Failure surface factor: risk based on ticket content and tags.

    Analyzes validator, test, and replay keywords plus critical tags.
    """
    title_lower = ticket.title.lower()
    desc_lower = ticket.description.lower()
    combined = f"{title_lower} {desc_lower}"

    validator_keywords = ("validator", "validation", "verify", "check", "audit")
    test_keywords = ("test", "spec", "integration", "e2e", "unit", "coverage")
    replay_keywords = ("replay", "log", "trace", "audit", "history", "rollback")

    validator_hits = sum(1 for kw in validator_keywords if kw in combined)
    test_hits = sum(1 for kw in test_keywords if kw in combined)
    replay_hits = sum(1 for kw in replay_keywords if kw in combined)

    validator_impact = min(validator_hits * 0.2, 1.0)
    test_impact = min(test_hits * 0.25, 1.0)
    replay_impact = min(replay_hits * 0.3, 1.0)

    # Priority multiplier
    priority_str = ticket.priority.lower()
    multiplier = 1.0
    if "critical" in priority_str:
        multiplier = 1.3
    elif "high" in priority_str:
        multiplier = 1.1
    elif "low" in priority_str:
        multiplier = 0.8

    base = validator_impact * 0.40 + test_impact * 0.35 + replay_impact * 0.25
    score = base * multiplier

    # Tag boost
    critical_tags = frozenset(
        {"security", "data-integrity", "performance", "compliance"}
    )
    tag_boost = sum(
        0.1 for tag in ticket.tags if any(ct in tag.lower() for ct in critical_tags)
    )
    score += min(tag_boost, 0.3)

    return max(0.0, min(1.0, score))


def _calculate_time_decay(ticket: ModelTicketData) -> float:
    """Time decay factor: older tickets get higher priority (anti-starvation).

    Uses exponential decay with 30-day half-life.
    Tickets >90 days old get 1.2x boost.
    """
    if ticket.created_at is None:
        return 0.5

    now = datetime.now(
        tz=ticket.created_at.tzinfo if ticket.created_at.tzinfo else None
    )
    if now.tzinfo is None and ticket.created_at.tzinfo is not None:
        now = datetime.now(tz=UTC)
    elif now.tzinfo is not None and ticket.created_at.tzinfo is None:
        now = datetime.now()

    age_days = max(0, (now - ticket.created_at).days)

    # Exponential decay with 30-day half-life (score increases with age)
    decay_factor = 0.5 ** (age_days / 30.0)
    time_score = 1.0 - decay_factor

    # Boost for very old tickets
    if age_days > 90:
        time_score = min(1.0, time_score * 1.2)

    return max(0.0, min(1.0, time_score))


def _calculate_agent_utility(
    requests: list[ModelAgentRequestData],
) -> float:
    """Agent utility factor: weighted agent request frequency.

    Higher when multiple unique agents request the ticket recently with boosts.
    """
    if not requests:
        return 0.0

    active = [r for r in requests if r.is_active]
    if not active:
        return 0.0

    unique_agents = {r.agent_id for r in active}
    total_boost = sum(r.priority_boost for r in active)
    avg_boost = total_boost / len(active)

    # Component scores
    diversity = min(len(unique_agents) * 0.2, 1.0)
    boost_score = min(avg_boost / 2.0, 1.0)
    request_factor = min(1.0, 1.0 + len(active) * 0.2)

    score = (diversity * 0.4 + boost_score * 0.3) * request_factor
    # Frequency bonus
    score += min(len(active) * 0.05, 0.2)

    return max(0.0, min(1.0, score))


def _calculate_user_weighting(
    ticket: ModelTicketData,
    overrides: list[ModelPlanOverrideData],
) -> float:
    """User weighting factor: manual priority overrides with time decay.

    Base score from ticket priority label, then override applied with
    exponential decay after 7 days.
    """
    priority_map = {
        "critical": 1.0,
        "high": 0.75,
        "medium": 0.5,
        "low": 0.25,
        "minimal": 0.1,
    }
    base_score = priority_map.get(ticket.priority.lower(), 0.5)

    # Apply most recent active override
    active = [o for o in overrides if o.is_active]
    if not active:
        return base_score

    # Sort by timestamp descending, pick most recent
    with_ts = [o for o in active if o.timestamp is not None]
    if not with_ts:
        return base_score

    def _ts(o: ModelPlanOverrideData) -> datetime:
        assert o.timestamp is not None  # guaranteed by with_ts filter
        return o.timestamp

    latest = max(with_ts, key=_ts)

    # Check expiry
    if latest.expires_at is not None:
        now = datetime.now(
            tz=latest.expires_at.tzinfo if latest.expires_at.tzinfo else None
        )
        if now > latest.expires_at:
            return base_score

    # Override score (0-100 -> 0-1)
    override_score = max(0.0, min(1.0, latest.override_score / 100.0))

    # Time decay: no decay in first 7 days, then exponential
    if latest.timestamp is not None:
        now = datetime.now(
            tz=latest.timestamp.tzinfo if latest.timestamp.tzinfo else None
        )
        days_old = max(0, (now - latest.timestamp).days)
        if days_old <= 7:
            return override_score
        excess = days_old - 7
        decay = 0.5 + 0.5 * (2.718 ** (-excess / 14.0))
        # Blend override with base as it ages
        return override_score * decay + base_score * (1.0 - decay)

    return override_score
