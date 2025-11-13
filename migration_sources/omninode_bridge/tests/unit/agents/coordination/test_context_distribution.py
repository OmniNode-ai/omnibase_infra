"""
Tests for agent context distribution system.

Validates context packaging, distribution, and updates with
performance requirements (<200ms per agent).
"""

import asyncio
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from omninode_bridge.agents.coordination.context_distribution import (
    ContextDistributor,
)
from omninode_bridge.agents.coordination.context_models import (
    AgentContext,
    CoordinationProtocols,
    ContextUpdateRequest,
    ResourceAllocation,
    SharedIntelligence,
)
from omninode_bridge.agents.coordination.thread_safe_state import ThreadSafeState
from omninode_bridge.agents.metrics.collector import MetricsCollector


class TestContextDistributor:
    """Test suite for ContextDistributor."""

    @pytest.fixture
    def state(self):
        """Create ThreadSafeState for tests."""
        return ThreadSafeState()

    @pytest.fixture
    def metrics_collector(self):
        """Create mock MetricsCollector for tests."""
        collector = MagicMock(spec=MetricsCollector)
        collector.record_timing = AsyncMock()
        collector.record_counter = AsyncMock()
        collector.record_gauge = AsyncMock()
        return collector

    @pytest.fixture
    def distributor(self, state, metrics_collector):
        """Create ContextDistributor for tests."""
        return ContextDistributor(state=state, metrics_collector=metrics_collector)

    @pytest.fixture
    def coordination_state(self):
        """Create coordination state for tests."""
        return {
            "coordination_id": "coord-test-123",
            "session_id": "session-test-456",
            "workflow_type": "code_generation",
        }

    @pytest.fixture
    def agent_assignments(self):
        """Create agent assignments for tests."""
        return {
            "model_gen": {
                "agent_role": "model_generator",
                "objective": "Generate Pydantic models from contract",
                "tasks": ["parse_contract", "generate_models", "validate_models"],
                "input_data": {
                    "contract_path": "./contract.yaml",
                    "output_dir": "./models",
                },
                "dependencies": [],
                "output_requirements": {
                    "format": "pydantic",
                    "version": "2.0",
                },
                "success_criteria": {
                    "min_quality": 0.9,
                    "all_fields_typed": True,
                },
            },
            "validator_gen": {
                "agent_role": "validator_generator",
                "objective": "Generate validators for models",
                "tasks": ["generate_validators", "generate_tests"],
                "input_data": {
                    "models_dir": "./models",
                },
                "dependencies": ["model_gen"],
                "output_requirements": {
                    "format": "pytest",
                },
                "success_criteria": {
                    "coverage": 0.95,
                },
            },
            "test_gen": {
                "agent_role": "test_generator",
                "objective": "Generate integration tests",
                "tasks": ["generate_integration_tests"],
                "input_data": {},
                "dependencies": ["model_gen", "validator_gen"],
                "output_requirements": {},
                "success_criteria": {},
            },
        }

    @pytest.fixture
    def shared_intelligence(self):
        """Create shared intelligence for tests."""
        return SharedIntelligence(
            type_registry={
                "UserId": "str",
                "Email": "str",
                "Timestamp": "datetime",
            },
            pattern_library={
                "validation": ["email_validator", "url_validator"],
                "serialization": ["json_serializer", "xml_serializer"],
            },
            validation_rules={
                "email": {"pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"},
                "url": {"pattern": "^https?://"},
            },
            naming_conventions={
                "class": "PascalCase",
                "function": "snake_case",
                "constant": "UPPER_SNAKE_CASE",
            },
            dependency_graph={
                "validator_gen": ["model_gen"],
                "test_gen": ["model_gen", "validator_gen"],
            },
        )

    # Basic functionality tests

    @pytest.mark.asyncio
    async def test_distribute_agent_context_basic(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test basic context distribution to multiple agents."""
        # Distribute contexts
        contexts = await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        # Verify all agents received contexts
        assert len(contexts) == 3
        assert "model_gen" in contexts
        assert "validator_gen" in contexts
        assert "test_gen" in contexts

        # Verify context structure
        for agent_id, context in contexts.items():
            assert isinstance(context, AgentContext)
            assert context.coordination_metadata.agent_id == agent_id
            assert context.coordination_metadata.coordination_id == "coord-test-123"
            assert context.coordination_metadata.session_id == "session-test-456"
            assert context.agent_assignment.objective != ""
            assert len(context.agent_assignment.tasks) > 0

    @pytest.mark.asyncio
    async def test_coordination_metadata_injection(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test coordination metadata is correctly injected."""
        contexts = await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        # Verify metadata for each agent
        model_gen_context = contexts["model_gen"]
        assert model_gen_context.coordination_metadata.agent_id == "model_gen"
        assert model_gen_context.coordination_metadata.agent_role == "model_generator"
        assert (
            model_gen_context.coordination_metadata.coordination_id == "coord-test-123"
        )
        assert model_gen_context.coordination_metadata.session_id == "session-test-456"
        assert isinstance(model_gen_context.coordination_metadata.created_at, datetime)

    @pytest.mark.asyncio
    async def test_shared_intelligence_distribution(
        self, distributor, coordination_state, agent_assignments, shared_intelligence
    ):
        """Test shared intelligence is distributed to all agents."""
        contexts = await distributor.distribute_agent_context(
            coordination_state=coordination_state,
            agent_assignments=agent_assignments,
            shared_intelligence=shared_intelligence,
        )

        # Verify all agents have same shared intelligence
        for context in contexts.values():
            assert context.shared_intelligence.type_registry == {
                "UserId": "str",
                "Email": "str",
                "Timestamp": "datetime",
            }
            assert "validation" in context.shared_intelligence.pattern_library
            assert "email" in context.shared_intelligence.validation_rules
            assert (
                context.shared_intelligence.naming_conventions["class"] == "PascalCase"
            )

    @pytest.mark.asyncio
    async def test_agent_assignment_distribution(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test agent assignments are correctly packaged."""
        contexts = await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        # Verify model_gen assignment
        model_gen_context = contexts["model_gen"]
        assert (
            model_gen_context.agent_assignment.objective
            == "Generate Pydantic models from contract"
        )
        assert "parse_contract" in model_gen_context.agent_assignment.tasks
        assert "generate_models" in model_gen_context.agent_assignment.tasks
        assert (
            model_gen_context.agent_assignment.input_data["contract_path"]
            == "./contract.yaml"
        )
        assert model_gen_context.agent_assignment.dependencies == []

        # Verify validator_gen assignment
        validator_gen_context = contexts["validator_gen"]
        assert "model_gen" in validator_gen_context.agent_assignment.dependencies

    @pytest.mark.asyncio
    async def test_resource_allocation_default(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test default resource allocation is applied."""
        contexts = await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        # Verify default resource allocation
        for context in contexts.values():
            assert context.resource_allocation.max_execution_time_ms == 300000
            assert context.resource_allocation.max_retry_attempts == 3
            assert context.resource_allocation.quality_threshold == 0.8
            assert context.resource_allocation.timeout_ms == 30000

    @pytest.mark.asyncio
    async def test_resource_allocation_custom(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test custom resource allocation per agent."""
        custom_allocations = {
            "model_gen": ResourceAllocation(
                max_execution_time_ms=60000,
                max_retry_attempts=5,
                quality_threshold=0.95,
            ),
            "validator_gen": ResourceAllocation(
                max_execution_time_ms=45000,
                max_retry_attempts=2,
                quality_threshold=0.85,
            ),
        }

        contexts = await distributor.distribute_agent_context(
            coordination_state=coordination_state,
            agent_assignments=agent_assignments,
            resource_allocations=custom_allocations,
        )

        # Verify custom allocations
        assert contexts["model_gen"].resource_allocation.max_execution_time_ms == 60000
        assert contexts["model_gen"].resource_allocation.max_retry_attempts == 5
        assert contexts["model_gen"].resource_allocation.quality_threshold == 0.95

        assert (
            contexts["validator_gen"].resource_allocation.max_execution_time_ms == 45000
        )
        assert contexts["validator_gen"].resource_allocation.max_retry_attempts == 2

        # test_gen should use default
        assert (
            contexts["test_gen"].resource_allocation.max_execution_time_ms == 300000
        )

    @pytest.mark.asyncio
    async def test_coordination_protocols_custom(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test custom coordination protocols per agent."""
        custom_protocols = {
            "model_gen": CoordinationProtocols(
                update_interval_ms=2000,
                heartbeat_interval_ms=5000,
                status_update_channel="kafka",
            ),
        }

        contexts = await distributor.distribute_agent_context(
            coordination_state=coordination_state,
            agent_assignments=agent_assignments,
            coordination_protocols=custom_protocols,
        )

        # Verify custom protocols
        assert contexts["model_gen"].coordination_protocols.update_interval_ms == 2000
        assert (
            contexts["model_gen"].coordination_protocols.heartbeat_interval_ms == 5000
        )
        assert (
            contexts["model_gen"].coordination_protocols.status_update_channel
            == "kafka"
        )

        # Others should use default
        assert (
            contexts["validator_gen"].coordination_protocols.update_interval_ms == 5000
        )

    # Context retrieval tests

    @pytest.mark.asyncio
    async def test_get_agent_context(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test retrieving agent context."""
        # Distribute contexts
        await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        # Retrieve context
        context = distributor.get_agent_context("coord-test-123", "model_gen")

        assert context is not None
        assert context.coordination_metadata.agent_id == "model_gen"
        assert context.coordination_metadata.coordination_id == "coord-test-123"

    @pytest.mark.asyncio
    async def test_get_agent_context_not_found(self, distributor):
        """Test retrieving non-existent agent context."""
        context = distributor.get_agent_context("coord-invalid", "agent-invalid")
        assert context is None

    @pytest.mark.asyncio
    async def test_list_coordination_contexts(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test listing all agent IDs for coordination."""
        # Distribute contexts
        await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        # List contexts
        agent_ids = distributor.list_coordination_contexts("coord-test-123")

        assert len(agent_ids) == 3
        assert "model_gen" in agent_ids
        assert "validator_gen" in agent_ids
        assert "test_gen" in agent_ids

    # Context update tests

    @pytest.mark.asyncio
    async def test_update_shared_intelligence_type_registry(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test updating type registry in shared intelligence."""
        # Distribute contexts
        await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        # Update type registry
        update_request = ContextUpdateRequest(
            coordination_id="coord-test-123",
            update_type="type_registry",
            update_data={"NewType": "CustomClass"},
            increment_version=True,
        )

        results = distributor.update_shared_intelligence(update_request)

        # Verify all updates succeeded
        assert len(results) == 3
        assert all(results.values())

        # Verify updated contexts
        for agent_id in ["model_gen", "validator_gen", "test_gen"]:
            context = distributor.get_agent_context("coord-test-123", agent_id)
            assert "NewType" in context.shared_intelligence.type_registry
            assert context.shared_intelligence.type_registry["NewType"] == "CustomClass"
            assert context.context_version == 2  # Incremented

    @pytest.mark.asyncio
    async def test_update_shared_intelligence_pattern_library(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test updating pattern library in shared intelligence."""
        # Distribute contexts
        await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        # Update pattern library
        update_request = ContextUpdateRequest(
            coordination_id="coord-test-123",
            update_type="pattern_library",
            update_data={"caching": ["redis_cache", "memory_cache"]},
            increment_version=True,
        )

        results = distributor.update_shared_intelligence(update_request)

        assert all(results.values())

        # Verify updated contexts
        context = distributor.get_agent_context("coord-test-123", "model_gen")
        assert "caching" in context.shared_intelligence.pattern_library
        assert "redis_cache" in context.shared_intelligence.pattern_library["caching"]

    @pytest.mark.asyncio
    async def test_update_shared_intelligence_target_agents(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test updating shared intelligence for specific agents only."""
        # Distribute contexts
        await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        # Update only model_gen and validator_gen
        update_request = ContextUpdateRequest(
            coordination_id="coord-test-123",
            update_type="type_registry",
            update_data={"SpecialType": "SpecialClass"},
            target_agents=["model_gen", "validator_gen"],
            increment_version=True,
        )

        results = distributor.update_shared_intelligence(update_request)

        assert results["model_gen"] is True
        assert results["validator_gen"] is True
        assert "test_gen" not in results  # Not targeted

        # Verify only targeted agents were updated
        model_gen_context = distributor.get_agent_context("coord-test-123", "model_gen")
        assert "SpecialType" in model_gen_context.shared_intelligence.type_registry

        test_gen_context = distributor.get_agent_context("coord-test-123", "test_gen")
        assert "SpecialType" not in test_gen_context.shared_intelligence.type_registry

    @pytest.mark.asyncio
    async def test_update_shared_intelligence_version_control(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test version control during updates."""
        # Distribute contexts
        await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        # Update without incrementing version
        update_request = ContextUpdateRequest(
            coordination_id="coord-test-123",
            update_type="type_registry",
            update_data={"Type1": "Class1"},
            increment_version=False,
        )

        distributor.update_shared_intelligence(update_request)

        context = distributor.get_agent_context("coord-test-123", "model_gen")
        assert context.context_version == 1  # Not incremented

        # Update with incrementing version
        update_request2 = ContextUpdateRequest(
            coordination_id="coord-test-123",
            update_type="type_registry",
            update_data={"Type2": "Class2"},
            increment_version=True,
        )

        distributor.update_shared_intelligence(update_request2)

        context = distributor.get_agent_context("coord-test-123", "model_gen")
        assert context.context_version == 2  # Incremented

    # Cleanup tests

    @pytest.mark.asyncio
    async def test_clear_coordination_contexts(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test clearing contexts for coordination workflow."""
        # Distribute contexts
        await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        # Verify contexts exist
        agent_ids = distributor.list_coordination_contexts("coord-test-123")
        assert len(agent_ids) == 3

        # Clear contexts
        cleared = distributor.clear_coordination_contexts("coord-test-123")
        assert cleared is True

        # Verify contexts are cleared
        agent_ids = distributor.list_coordination_contexts("coord-test-123")
        assert len(agent_ids) == 0

    @pytest.mark.asyncio
    async def test_clear_coordination_contexts_not_found(self, distributor):
        """Test clearing non-existent coordination contexts."""
        cleared = distributor.clear_coordination_contexts("coord-invalid")
        assert cleared is False

    # Performance tests

    @pytest.mark.asyncio
    async def test_distribution_performance_target(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test distribution meets <200ms per agent target."""
        import time

        start_time = time.time()

        await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        elapsed_ms = (time.time() - start_time) * 1000
        per_agent_ms = elapsed_ms / len(agent_assignments)

        # Performance target: <200ms per agent
        assert per_agent_ms < 200, f"Distribution took {per_agent_ms:.2f}ms per agent"

    @pytest.mark.asyncio
    async def test_distribution_performance_50_agents(self, distributor):
        """Test distribution scales to 50+ agents."""
        import time

        # Create 50 agent assignments
        agent_assignments = {
            f"agent_{i}": {
                "agent_role": f"role_{i}",
                "objective": f"Objective {i}",
                "tasks": [f"task_{i}_1", f"task_{i}_2"],
                "input_data": {"data": f"value_{i}"},
                "dependencies": [],
                "output_requirements": {},
                "success_criteria": {},
            }
            for i in range(50)
        }

        coordination_state = {
            "coordination_id": "coord-scale-test",
            "session_id": "session-scale-test",
        }

        start_time = time.time()

        contexts = await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        elapsed_ms = (time.time() - start_time) * 1000
        per_agent_ms = elapsed_ms / len(agent_assignments)

        # Verify all agents received contexts
        assert len(contexts) == 50

        # Performance target: <200ms per agent
        assert per_agent_ms < 200, f"Distribution took {per_agent_ms:.2f}ms per agent"

    @pytest.mark.asyncio
    async def test_context_retrieval_performance(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test context retrieval is fast (<5ms)."""
        import time

        # Distribute contexts
        await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        # Measure retrieval time
        start_time = time.time()

        context = distributor.get_agent_context("coord-test-123", "model_gen")

        elapsed_ms = (time.time() - start_time) * 1000

        assert context is not None
        # Performance target: <5ms
        assert elapsed_ms < 5, f"Retrieval took {elapsed_ms:.2f}ms"

    # Thread safety tests

    @pytest.mark.asyncio
    async def test_concurrent_distribution(
        self, state, metrics_collector, agent_assignments
    ):
        """Test concurrent context distribution is thread-safe."""
        distributor = ContextDistributor(state=state, metrics_collector=metrics_collector)

        # Create multiple coordination states
        coordination_states = [
            {
                "coordination_id": f"coord-concurrent-{i}",
                "session_id": f"session-concurrent-{i}",
            }
            for i in range(10)
        ]

        # Distribute contexts concurrently
        tasks = [
            distributor.distribute_agent_context(
                coordination_state=coord_state, agent_assignments=agent_assignments
            )
            for coord_state in coordination_states
        ]

        results = await asyncio.gather(*tasks)

        # Verify all distributions succeeded
        assert len(results) == 10
        for contexts in results:
            assert len(contexts) == 3

    @pytest.mark.asyncio
    async def test_concurrent_updates(
        self, distributor, coordination_state, agent_assignments
    ):
        """Test concurrent shared intelligence updates are thread-safe."""
        # Distribute contexts
        await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        # Create multiple update requests
        update_requests = [
            ContextUpdateRequest(
                coordination_id="coord-test-123",
                update_type="type_registry",
                update_data={f"Type{i}": f"Class{i}"},
                increment_version=True,
            )
            for i in range(20)
        ]

        # Apply updates concurrently (synchronous, but simulate concurrent access)
        results = [
            distributor.update_shared_intelligence(req) for req in update_requests
        ]

        # Verify all updates succeeded
        assert len(results) == 20
        for result in results:
            assert all(result.values())

        # Verify final version
        context = distributor.get_agent_context("coord-test-123", "model_gen")
        assert context.context_version == 21  # 1 + 20 updates

    # Error handling tests

    @pytest.mark.asyncio
    async def test_distribution_missing_coordination_id(
        self, distributor, agent_assignments
    ):
        """Test distribution fails gracefully with missing coordination_id."""
        with pytest.raises(ValueError, match="coordination_id"):
            await distributor.distribute_agent_context(
                coordination_state={"session_id": "session-123"},
                agent_assignments=agent_assignments,
            )

    @pytest.mark.asyncio
    async def test_distribution_missing_session_id(
        self, distributor, agent_assignments
    ):
        """Test distribution fails gracefully with missing session_id."""
        with pytest.raises(ValueError, match="session_id"):
            await distributor.distribute_agent_context(
                coordination_state={"coordination_id": "coord-123"},
                agent_assignments=agent_assignments,
            )

    @pytest.mark.asyncio
    async def test_update_shared_intelligence_invalid_coordination(self, distributor):
        """Test update fails gracefully for invalid coordination_id."""
        update_request = ContextUpdateRequest(
            coordination_id="coord-invalid",
            update_type="type_registry",
            update_data={"Type": "Class"},
        )

        results = distributor.update_shared_intelligence(update_request)

        assert len(results) == 0  # No agents to update

    # Metrics tests

    @pytest.mark.asyncio
    async def test_metrics_recording(
        self, distributor, coordination_state, agent_assignments, metrics_collector
    ):
        """Test metrics are recorded during distribution."""
        await distributor.distribute_agent_context(
            coordination_state=coordination_state, agent_assignments=agent_assignments
        )

        # Verify timing metrics were recorded
        metrics_collector.record_timing.assert_any_call(
            metric_name="context_distribution_time_ms",
            duration_ms=pytest.approx(0, abs=500),  # Should be fast
            tags={
                "coordination_id": "coord-test-123",
                "agent_count": "3",
            },
            correlation_id="coord-test-123",
        )

        # Verify per-agent timing was recorded
        metrics_collector.record_timing.assert_any_call(
            metric_name="context_distribution_per_agent_ms",
            duration_ms=pytest.approx(0, abs=200),  # Should be <200ms
            tags={"coordination_id": "coord-test-123"},
            correlation_id="coord-test-123",
        )

        # Verify gauge metrics for context size
        assert metrics_collector.record_gauge.call_count >= 3  # One per agent

    # Integration tests

    @pytest.mark.asyncio
    async def test_code_generation_workflow_context(
        self, distributor, shared_intelligence
    ):
        """Test context distribution for code generation workflow."""
        coordination_state = {
            "coordination_id": "codegen-workflow-1",
            "session_id": "codegen-session-1",
        }

        agent_assignments = {
            "model_generator": {
                "agent_role": "model_generator",
                "objective": "Generate Pydantic models from contract",
                "tasks": ["parse_contract", "generate_models"],
                "input_data": {"contract_path": "./contract.yaml"},
                "dependencies": [],
                "output_requirements": {"format": "pydantic_v2"},
                "success_criteria": {"quality": 0.95},
            },
            "validator_generator": {
                "agent_role": "validator_generator",
                "objective": "Generate validators",
                "tasks": ["generate_validators"],
                "input_data": {"models_context": "from_previous_agent"},
                "dependencies": ["model_generator"],
                "output_requirements": {},
                "success_criteria": {},
            },
            "test_generator": {
                "agent_role": "test_generator",
                "objective": "Generate tests",
                "tasks": ["generate_tests"],
                "input_data": {},
                "dependencies": ["model_generator", "validator_generator"],
                "output_requirements": {},
                "success_criteria": {},
            },
        }

        # Distribute with shared intelligence
        contexts = await distributor.distribute_agent_context(
            coordination_state=coordination_state,
            agent_assignments=agent_assignments,
            shared_intelligence=shared_intelligence,
        )

        # Verify workflow-specific context
        assert len(contexts) == 3

        # Verify model_generator has no dependencies
        model_gen = contexts["model_generator"]
        assert len(model_gen.agent_assignment.dependencies) == 0
        assert model_gen.shared_intelligence.type_registry["UserId"] == "str"

        # Verify validator_generator depends on model_generator
        validator_gen = contexts["validator_generator"]
        assert "model_generator" in validator_gen.agent_assignment.dependencies

        # Verify test_generator depends on both
        test_gen = contexts["test_generator"]
        assert "model_generator" in test_gen.agent_assignment.dependencies
        assert "validator_generator" in test_gen.agent_assignment.dependencies
