"""Unit tests for CapabilityMatchEngine."""

from datetime import datetime

import pytest

from omninode_bridge.agents.registry.matcher import CapabilityMatchEngine
from omninode_bridge.agents.registry.models import (
    AgentInfo,
    AgentMetadata,
    AgentStatus,
    AgentType,
    Task,
)


@pytest.fixture
def matcher():
    """Create CapabilityMatchEngine with default weights."""
    return CapabilityMatchEngine()


@pytest.fixture
def sample_agent():
    """Create sample agent for testing."""
    metadata = AgentMetadata(
        agent_type=AgentType.CONTRACT_INFERENCER,
        version="1.0.0",
        description="Test agent",
        priority=50,
        max_concurrent_tasks=10,
        success_rate=0.8,
    )

    return AgentInfo(
        agent_id="test_agent",
        capabilities=["contract_inference", "yaml_parsing"],
        metadata=metadata,
        registered_at=datetime.utcnow(),
        last_heartbeat=datetime.utcnow(),
        status=AgentStatus.ACTIVE,
        active_tasks=0,
    )


@pytest.fixture
def sample_task():
    """Create sample task for testing."""
    return Task(
        task_type="contract_inference",
        required_capabilities=["contract_inference", "yaml_parsing"],
    )


class TestCapabilityScoring:
    """Tests for capability matching score."""

    def test_perfect_capability_match(self, matcher, sample_agent, sample_task):
        """Test perfect capability match gives high score."""
        score = matcher.score_agent(sample_agent, sample_task)

        # Perfect capability match should give 1.0 for capability score
        assert score.capability_score == 1.0
        assert score.total > 0.5

    def test_partial_capability_match(self, matcher, sample_task):
        """Test partial capability match gives lower score."""
        metadata = AgentMetadata(
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test agent",
            priority=50,
            max_concurrent_tasks=10,
        )

        # Agent with only one matching capability
        agent = AgentInfo(
            agent_id="partial_agent",
            capabilities=["contract_inference"],  # Missing yaml_parsing
            metadata=metadata,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            status=AgentStatus.ACTIVE,
        )

        score = matcher.score_agent(agent, sample_task)

        # Partial match should have lower capability score
        assert 0.0 < score.capability_score < 1.0

    def test_no_capability_match(self, matcher, sample_task):
        """Test no capability match gives low score."""
        metadata = AgentMetadata(
            agent_type=AgentType.VALIDATOR,
            version="1.0.0",
            description="Test agent",
            priority=50,
            max_concurrent_tasks=10,
        )

        # Agent with no matching capabilities
        agent = AgentInfo(
            agent_id="no_match_agent",
            capabilities=["validation", "testing"],
            metadata=metadata,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            status=AgentStatus.ACTIVE,
        )

        score = matcher.score_agent(agent, sample_task)

        # No match should have 0.0 capability score
        assert score.capability_score == 0.0

    def test_task_with_no_required_capabilities(self, matcher, sample_agent):
        """Test task with no required capabilities matches any agent."""
        task = Task(task_type="generic", required_capabilities=[])

        score = matcher.score_agent(sample_agent, task)

        # No requirements should give perfect capability match
        assert score.capability_score == 1.0


class TestLoadScoring:
    """Tests for load balance scoring."""

    def test_unloaded_agent(self, matcher, sample_task):
        """Test unloaded agent gets high load score."""
        metadata = AgentMetadata(
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test agent",
            max_concurrent_tasks=10,
        )

        agent = AgentInfo(
            agent_id="unloaded_agent",
            capabilities=["contract_inference"],
            metadata=metadata,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            status=AgentStatus.ACTIVE,
            active_tasks=0,  # No active tasks
        )

        score = matcher.score_agent(agent, sample_task)

        # No load should give 1.0 load score
        assert score.load_score == 1.0

    def test_fully_loaded_agent(self, matcher, sample_task):
        """Test fully loaded agent gets low load score."""
        metadata = AgentMetadata(
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test agent",
            max_concurrent_tasks=10,
        )

        agent = AgentInfo(
            agent_id="loaded_agent",
            capabilities=["contract_inference"],
            metadata=metadata,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            status=AgentStatus.ACTIVE,
            active_tasks=10,  # At max capacity
        )

        score = matcher.score_agent(agent, sample_task)

        # Full load should give 0.0 load score
        assert score.load_score == 0.0

    def test_half_loaded_agent(self, matcher, sample_task):
        """Test half-loaded agent gets medium load score."""
        metadata = AgentMetadata(
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test agent",
            max_concurrent_tasks=10,
        )

        agent = AgentInfo(
            agent_id="half_loaded_agent",
            capabilities=["contract_inference"],
            metadata=metadata,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            status=AgentStatus.ACTIVE,
            active_tasks=5,  # Half capacity
        )

        score = matcher.score_agent(agent, sample_task)

        # Half load should give 0.5 load score
        assert score.load_score == 0.5


class TestPriorityScoring:
    """Tests for priority scoring."""

    def test_high_priority_agent(self, matcher, sample_task):
        """Test high priority agent gets high priority score."""
        metadata = AgentMetadata(
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test agent",
            priority=100,  # Maximum priority
        )

        agent = AgentInfo(
            agent_id="high_priority_agent",
            capabilities=["contract_inference"],
            metadata=metadata,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            status=AgentStatus.ACTIVE,
        )

        score = matcher.score_agent(agent, sample_task)

        # Priority 100 should give 1.0 priority score
        assert score.priority_score == 1.0

    def test_low_priority_agent(self, matcher, sample_task):
        """Test low priority agent gets low priority score."""
        metadata = AgentMetadata(
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test agent",
            priority=0,  # Minimum priority
        )

        agent = AgentInfo(
            agent_id="low_priority_agent",
            capabilities=["contract_inference"],
            metadata=metadata,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            status=AgentStatus.ACTIVE,
        )

        score = matcher.score_agent(agent, sample_task)

        # Priority 0 should give 0.0 priority score
        assert score.priority_score == 0.0


class TestSuccessRateScoring:
    """Tests for success rate scoring."""

    def test_high_success_rate(self, matcher, sample_task):
        """Test agent with high success rate gets high score."""
        metadata = AgentMetadata(
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test agent",
            success_rate=1.0,  # Perfect success rate
        )

        agent = AgentInfo(
            agent_id="perfect_agent",
            capabilities=["contract_inference"],
            metadata=metadata,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            status=AgentStatus.ACTIVE,
        )

        score = matcher.score_agent(agent, sample_task)

        # Perfect success rate should give 1.0 success rate score
        assert score.success_rate_score == 1.0

    def test_no_success_rate(self, matcher, sample_task):
        """Test agent with no success rate gets default score."""
        metadata = AgentMetadata(
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Test agent",
            success_rate=None,  # No success rate data
        )

        agent = AgentInfo(
            agent_id="new_agent",
            capabilities=["contract_inference"],
            metadata=metadata,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            status=AgentStatus.ACTIVE,
        )

        score = matcher.score_agent(agent, sample_task)

        # No success rate should default to 0.5
        assert score.success_rate_score == 0.5


class TestCustomWeights:
    """Tests for custom weight configuration."""

    def test_custom_weights(self, sample_agent, sample_task):
        """Test matcher with custom weights."""
        custom_matcher = CapabilityMatchEngine(
            weights={
                "capability": 0.5,
                "load": 0.2,
                "priority": 0.2,
                "success_rate": 0.1,
            }
        )

        score = custom_matcher.score_agent(sample_agent, sample_task)

        # Score should be within valid range
        assert 0.0 <= score.total <= 1.0

        # Verify individual scores exist
        assert score.capability_score >= 0.0
        assert score.load_score >= 0.0
        assert score.priority_score >= 0.0
        assert score.success_rate_score >= 0.0


class TestScoreExplanation:
    """Tests for score explanation generation."""

    def test_explanation_includes_agent_id(self, matcher, sample_agent, sample_task):
        """Test explanation includes agent ID."""
        score = matcher.score_agent(sample_agent, sample_task)
        assert sample_agent.agent_id in score.explanation

    def test_explanation_includes_scores(self, matcher, sample_agent, sample_task):
        """Test explanation includes all scores."""
        score = matcher.score_agent(sample_agent, sample_task)

        # Explanation should mention key components
        explanation_lower = score.explanation.lower()
        assert "load" in explanation_lower
        assert "priority" in explanation_lower
        assert "success" in explanation_lower


class TestComprehensiveScoring:
    """Comprehensive tests for overall scoring."""

    def test_total_score_range(self, matcher, sample_agent, sample_task):
        """Test total score is always in valid range."""
        score = matcher.score_agent(sample_agent, sample_task)
        assert 0.0 <= score.total <= 1.0

    def test_score_comparison(self, matcher, sample_task):
        """Test scoring correctly ranks agents."""
        # High-quality agent
        high_quality_metadata = AgentMetadata(
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="High quality agent",
            priority=90,
            max_concurrent_tasks=10,
            success_rate=0.95,
        )
        high_quality_agent = AgentInfo(
            agent_id="high_quality",
            capabilities=["contract_inference", "yaml_parsing"],
            metadata=high_quality_metadata,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            status=AgentStatus.ACTIVE,
            active_tasks=0,
        )

        # Low-quality agent
        low_quality_metadata = AgentMetadata(
            agent_type=AgentType.CONTRACT_INFERENCER,
            version="1.0.0",
            description="Low quality agent",
            priority=10,
            max_concurrent_tasks=10,
            success_rate=0.50,
        )
        low_quality_agent = AgentInfo(
            agent_id="low_quality",
            capabilities=["contract_inference"],  # Missing yaml_parsing
            metadata=low_quality_metadata,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            status=AgentStatus.ACTIVE,
            active_tasks=5,  # Half loaded
        )

        high_score = matcher.score_agent(high_quality_agent, sample_task)
        low_score = matcher.score_agent(low_quality_agent, sample_task)

        # High quality agent should score higher
        assert high_score.total > low_score.total
