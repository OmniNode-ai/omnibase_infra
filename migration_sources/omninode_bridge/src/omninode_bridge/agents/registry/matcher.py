"""Capability matching engine for agent selection."""

from typing import Optional

from .models import AgentInfo, ConfidenceScore, Task


class CapabilityMatchEngine:
    """
    Capability matching engine with multi-criteria scoring.

    Scoring Algorithm:
    - Capability Match: 40% weight (Jaccard similarity)
    - Load Balance: 20% weight (1 - active_tasks / max_tasks)
    - Priority: 20% weight (agent priority / 100)
    - Success Rate: 20% weight (success_rate)

    Total Score: weighted sum (0.0-1.0)

    Example:
        ```python
        matcher = CapabilityMatchEngine()

        # Default weights
        score = matcher.score_agent(agent, task)
        print(f"Confidence: {score.total:.2f}")
        print(f"Explanation: {score.explanation}")

        # Custom weights
        matcher = CapabilityMatchEngine(weights={
            "capability": 0.5,
            "load": 0.2,
            "priority": 0.2,
            "success_rate": 0.1
        })
        ```
    """

    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        """
        Initialize capability matcher.

        Args:
            weights: Custom weights for scoring criteria.
                     Default: {"capability": 0.4, "load": 0.2, "priority": 0.2, "success_rate": 0.2}
        """
        self.weights = weights or {
            "capability": 0.4,
            "load": 0.2,
            "priority": 0.2,
            "success_rate": 0.2,
        }

    def score_agent(self, agent: AgentInfo, task: Task) -> ConfidenceScore:
        """
        Score agent suitability for task with detailed breakdown.

        Args:
            agent: Agent to score
            task: Task to match

        Returns:
            ConfidenceScore with detailed breakdown and explanation
        """
        # 1. Capability matching (Jaccard similarity)
        capability_score = self._score_capabilities(agent, task)

        # 2. Load balancing
        load_score = self._score_load(agent)

        # 3. Priority match
        priority_score = agent.metadata.priority / 100.0

        # 4. Success rate
        success_rate_score = agent.metadata.success_rate or 0.5

        # 5. Calculate weighted total
        total_score = (
            self.weights["capability"] * capability_score
            + self.weights["load"] * load_score
            + self.weights["priority"] * priority_score
            + self.weights["success_rate"] * success_rate_score
        )

        # Generate explanation
        explanation = self._generate_explanation(
            agent,
            task,
            capability_score,
            load_score,
            priority_score,
            success_rate_score,
        )

        return ConfidenceScore(
            total=total_score,
            capability_score=capability_score,
            load_score=load_score,
            priority_score=priority_score,
            success_rate_score=success_rate_score,
            explanation=explanation,
        )

    def _score_capabilities(self, agent: AgentInfo, task: Task) -> float:
        """
        Score capability match using Jaccard similarity.

        Jaccard similarity = |A & B| / |A | B|

        Args:
            agent: Agent to score
            task: Task to match

        Returns:
            Capability score (0.0-1.0)
        """
        agent_caps = set(agent.capabilities)
        task_caps = set(task.required_capabilities)

        if not task_caps:
            # No specific requirements - all agents match
            return 1.0

        intersection = len(agent_caps & task_caps)
        union = len(agent_caps | task_caps)

        return intersection / union if union > 0 else 0.0

    def _score_load(self, agent: AgentInfo) -> float:
        """
        Score load balance (prefer agents with lower load).

        Args:
            agent: Agent to score

        Returns:
            Load score (0.0-1.0, higher = less loaded)
        """
        max_tasks = agent.metadata.max_concurrent_tasks
        active_tasks = agent.active_tasks

        if max_tasks <= 0:
            return 0.0

        return 1.0 - (active_tasks / max_tasks)

    def _generate_explanation(
        self,
        agent: AgentInfo,
        task: Task,
        cap_score: float,
        load_score: float,
        pri_score: float,
        sr_score: float,
    ) -> str:
        """
        Generate human-readable explanation for scoring.

        Args:
            agent: Agent being scored
            task: Task being matched
            cap_score: Capability match score
            load_score: Load balance score
            pri_score: Priority score
            sr_score: Success rate score

        Returns:
            Human-readable explanation string
        """
        parts = [
            f"Agent '{agent.agent_id}' scored {cap_score:.2f} for capability match",
            f"Load: {load_score:.2f} ({agent.active_tasks}/{agent.metadata.max_concurrent_tasks} tasks)",
            f"Priority: {pri_score:.2f}",
            f"Success Rate: {sr_score:.2f}",
        ]

        return " | ".join(parts)
