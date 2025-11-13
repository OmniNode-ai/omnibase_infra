"""
Contract validation tests for ONEX v2.0 compliance.

Validates:
- YAML syntax and structure
- ONEX v2.0 schema compliance
- Subcontract references
- Event topic references
- Performance requirements specification
"""

from pathlib import Path

import pytest
import yaml

# Contract file paths
ORCHESTRATOR_CONTRACT = Path(
    "/Volumes/PRO-G40/Code/omninode_bridge/src/omninode_bridge/nodes/"
    "codegen_orchestrator/v1_0_0/contract.yaml"
)
METRICS_REDUCER_CONTRACT = Path(
    "/Volumes/PRO-G40/Code/omninode_bridge/src/omninode_bridge/nodes/"
    "codegen_metrics_reducer/v1_0_0/contract.yaml"
)


@pytest.fixture
def orchestrator_contract():
    """Load orchestrator contract YAML."""
    with open(ORCHESTRATOR_CONTRACT) as f:
        return yaml.safe_load(f)


@pytest.fixture
def metrics_reducer_contract():
    """Load metrics reducer contract YAML."""
    with open(METRICS_REDUCER_CONTRACT) as f:
        return yaml.safe_load(f)


def load_subcontracts(contract_path: Path, contract_data: dict) -> dict:
    """
    Load external subcontract files and merge them into the contract.

    Args:
        contract_path: Path to the main contract file
        contract_data: The main contract data dictionary

    Returns:
        Contract data with external subcontracts loaded and merged
    """
    if "subcontracts" in contract_data and "refs" in contract_data["subcontracts"]:
        contract_dir = contract_path.parent
        loaded_subcontracts = {}

        for ref_path in contract_data["subcontracts"]["refs"]:
            # Resolve relative path from contract directory
            subcontract_file = contract_dir / ref_path

            # Extract subcontract name from filename (e.g., "workflow.yaml" -> "workflow_coordination")
            filename = subcontract_file.stem
            if filename == "workflow":
                key = "workflow_coordination"
            elif filename == "events":
                key = "event_type"
            elif filename == "aggregation":
                key = "aggregation"
            elif filename == "streaming":
                key = "streaming"
            else:
                key = filename

            # Load subcontract if file exists
            if subcontract_file.exists():
                with open(subcontract_file) as f:
                    loaded_subcontracts[key] = yaml.safe_load(f)

        # Merge loaded subcontracts back into main contract
        contract_data["subcontracts"].update(loaded_subcontracts)

    return contract_data


# ============================================================================
# YAML SYNTAX TESTS
# ============================================================================


def test_orchestrator_contract_valid_yaml():
    """Test that orchestrator contract is valid YAML."""
    try:
        with open(ORCHESTRATOR_CONTRACT) as f:
            yaml.safe_load(f)
    except yaml.YAMLError as e:
        pytest.fail(f"Orchestrator contract has invalid YAML: {e}")


def test_metrics_reducer_contract_valid_yaml():
    """Test that metrics reducer contract is valid YAML."""
    try:
        with open(METRICS_REDUCER_CONTRACT) as f:
            yaml.safe_load(f)
    except yaml.YAMLError as e:
        pytest.fail(f"Metrics reducer contract has invalid YAML: {e}")


# ============================================================================
# ONEX V2.0 SCHEMA COMPLIANCE TESTS
# ============================================================================


def test_orchestrator_contract_has_required_fields(orchestrator_contract):
    """Test that orchestrator contract has all required ONEX v2.0 fields."""
    required_fields = [
        "schema_version",
        "contract_version",
        "node_identity",
        "metadata",
        "performance_requirements",
        "input_state",
        "output_state",
        "subcontracts",
        "dependencies",
    ]

    for field in required_fields:
        assert field in orchestrator_contract, f"Missing required field: {field}"


def test_metrics_reducer_contract_has_required_fields(metrics_reducer_contract):
    """Test that metrics reducer contract has all required fields."""
    required_fields = [
        "metadata",
        "contract",
        "subcontracts",  # Changed: aggregation now in subcontracts
        "performance",
        "dependencies",
        "kafka",
    ]

    for field in required_fields:
        assert field in metrics_reducer_contract, f"Missing required field: {field}"

    # Verify subcontracts are properly referenced
    assert "refs" in metrics_reducer_contract["subcontracts"]
    subcontract_refs = metrics_reducer_contract["subcontracts"]["refs"]
    assert "./contracts/intent_publisher.yaml" in subcontract_refs
    assert "./contracts/aggregation.yaml" in subcontract_refs
    assert "./contracts/streaming.yaml" in subcontract_refs


def test_orchestrator_node_identity(orchestrator_contract):
    """Test orchestrator node identity fields."""
    node_identity = orchestrator_contract["node_identity"]

    assert "name" in node_identity
    assert "display_name" in node_identity
    assert "node_type" in node_identity
    assert "version" in node_identity

    assert node_identity["node_type"] == "orchestrator"
    assert node_identity["version"] == "1.0.0"


def test_metrics_reducer_metadata(metrics_reducer_contract):
    """Test metrics reducer metadata fields."""
    metadata = metrics_reducer_contract["metadata"]

    assert "name" in metadata
    assert "version" in metadata
    assert "node_type" in metadata

    assert metadata["node_type"] == "reducer"
    assert metadata["version"] == "1.0.0"


# ============================================================================
# SUBCONTRACT VALIDATION TESTS
# ============================================================================


def test_orchestrator_subcontracts_valid(orchestrator_contract):
    """Test that orchestrator subcontracts are valid."""
    # Load external subcontract files
    contract = load_subcontracts(ORCHESTRATOR_CONTRACT, orchestrator_contract)
    subcontracts = contract["subcontracts"]

    # Check workflow_coordination subcontract
    assert "workflow_coordination" in subcontracts
    workflow = subcontracts["workflow_coordination"]

    assert "type" in workflow
    assert "stages" in workflow
    assert len(workflow["stages"]) == 8  # 8 pipeline stages

    # Verify each stage has required fields
    for stage in workflow["stages"]:
        assert "name" in stage
        assert "description" in stage
        assert "target_duration_ms" in stage

    # Check event_type subcontract
    assert "event_type" in subcontracts
    event_type = subcontracts["event_type"]

    assert event_type["publisher"] is True
    assert event_type["consumer"] is True
    assert "topics" in event_type
    assert "publishes" in event_type["topics"]
    assert "consumes" in event_type["topics"]


def test_metrics_reducer_aggregation_subcontract(metrics_reducer_contract):
    """Test that metrics reducer aggregation subcontract is valid."""
    # Load external subcontract files
    contract = load_subcontracts(METRICS_REDUCER_CONTRACT, metrics_reducer_contract)
    aggregation = contract["subcontracts"]["aggregation"]

    assert "aggregation_type" in aggregation
    assert "window_type" in aggregation
    assert "batch_size" in aggregation
    assert "strategies" in aggregation

    # Verify strategies
    strategies = aggregation["strategies"]
    assert len(strategies) >= 1

    for strategy in strategies:
        assert "name" in strategy
        assert "description" in strategy


# ============================================================================
# EVENT REFERENCE VALIDATION TESTS
# ============================================================================


def test_all_event_references_exist(orchestrator_contract):
    """Test that all event topic references exist and are valid."""
    # Load external subcontract files
    contract = load_subcontracts(ORCHESTRATOR_CONTRACT, orchestrator_contract)
    event_type = contract["subcontracts"]["event_type"]

    # Validate publish topics
    publish_topics = event_type["topics"]["publishes"]
    expected_publish_topics = [
        "dev.omninode-bridge.codegen.generation-started.v1",
        "dev.omninode-bridge.codegen.stage-completed.v1",
        "dev.omninode-bridge.codegen.generation-completed.v1",
        "dev.omninode-bridge.codegen.generation-failed.v1",
        "dev.omniarchon.intelligence.query-requested.v1",
        "dev.omniarchon.intelligence.pattern-storage-requested.v1",
    ]

    for topic in expected_publish_topics:
        assert topic in publish_topics, f"Missing publish topic: {topic}"

    # Validate consume topics
    consume_topics = event_type["topics"]["consumes"]
    expected_consume_topics = [
        "dev.omninode-bridge.codegen.generation-requested.v1",
        "dev.omniarchon.intelligence.query-completed.v1",
        "dev.omninode-bridge.codegen.checkpoint-response.v1",
    ]

    for topic in expected_consume_topics:
        assert topic in consume_topics, f"Missing consume topic: {topic}"


def test_metrics_reducer_kafka_topics(metrics_reducer_contract):
    """Test that metrics reducer Kafka topics are valid."""
    kafka = metrics_reducer_contract["kafka"]

    # Validate consumer topics
    consumer_topics = kafka["consumer_topics"]
    expected_consumer_topics = [
        "dev.omninode-bridge.codegen.generation-started.v1",
        "dev.omninode-bridge.codegen.stage-completed.v1",
        "dev.omninode-bridge.codegen.generation-completed.v1",
        "dev.omninode-bridge.codegen.generation-failed.v1",
    ]

    for topic in expected_consumer_topics:
        assert topic in consumer_topics, f"Missing consumer topic: {topic}"

    # Validate producer topics
    producer_topics = kafka["producer_topics"]
    assert "dev.omninode-bridge.codegen.metrics-recorded.v1" in producer_topics


# ============================================================================
# PERFORMANCE REQUIREMENTS TESTS
# ============================================================================


def test_orchestrator_performance_requirements(orchestrator_contract):
    """Test that orchestrator performance requirements are specified."""
    perf = orchestrator_contract["performance_requirements"]

    # Execution time requirements
    assert "execution_time" in perf
    exec_time = perf["execution_time"]
    assert "target_ms" in exec_time
    assert "max_ms" in exec_time
    assert exec_time["target_ms"] == 53000  # 53 seconds
    assert exec_time["max_ms"] == 90000  # 90 seconds

    # Memory requirements
    assert "memory_usage" in perf
    memory = perf["memory_usage"]
    assert "target_mb" in memory
    assert "max_mb" in memory
    assert memory["target_mb"] == 512
    assert memory["max_mb"] == 1024

    # Throughput requirements
    assert "throughput" in perf
    throughput = perf["throughput"]
    assert "target_requests_per_second" in throughput
    assert "max_concurrent_workflows" in throughput
    assert throughput["target_requests_per_second"] == 2
    assert throughput["max_concurrent_workflows"] == 10


def test_metrics_reducer_performance_requirements(metrics_reducer_contract):
    """Test that metrics reducer performance requirements are specified."""
    perf = metrics_reducer_contract["performance"]

    # Throughput target
    assert "throughput_target" in perf
    throughput = perf["throughput_target"]
    assert "events_per_second" in throughput
    assert throughput["events_per_second"] == 1000

    # Latency target
    assert "latency_target" in perf
    latency = perf["latency_target"]
    assert "p95_ms" in latency
    assert latency["p95_ms"] == 100

    # Resource limits
    assert "resource_limits" in perf
    resources = perf["resource_limits"]
    assert "max_memory_mb" in resources
    assert "max_cpu_percent" in resources
    assert resources["max_memory_mb"] == 256


# ============================================================================
# DEPENDENCY VALIDATION TESTS
# ============================================================================


def test_orchestrator_dependencies(orchestrator_contract):
    """Test that orchestrator dependencies are properly specified."""
    deps = orchestrator_contract["dependencies"]

    # Service dependencies
    assert "services" in deps
    services = deps["services"]

    kafka_service = next((s for s in services if s["name"] == "kafka"), None)
    assert kafka_service is not None
    assert kafka_service["type"] == "event_bus"
    assert kafka_service["required"] is True

    omniarchon_service = next((s for s in services if s["name"] == "omniarchon"), None)
    assert omniarchon_service is not None
    assert omniarchon_service["type"] == "intelligence_service"
    assert omniarchon_service["required"] is False

    # Library dependencies
    assert "libraries" in deps
    libraries = deps["libraries"]

    llama_index = next(
        (lib for lib in libraries if lib["name"] == "llama-index-core"), None
    )
    assert llama_index is not None
    assert "version" in llama_index


def test_metrics_reducer_dependencies(metrics_reducer_contract):
    """Test that metrics reducer dependencies are properly specified."""
    deps = metrics_reducer_contract["dependencies"]

    # Required services
    assert "required_services" in deps
    required = deps["required_services"]

    kafka_service = next((s for s in required if s["name"] == "kafka"), None)
    assert kafka_service is not None
    assert kafka_service["type"] == "event_bus"

    # Optional services
    assert "optional_services" in deps
    optional = deps["optional_services"]

    postgres_service = next((s for s in optional if s["name"] == "postgresql"), None)
    assert postgres_service is not None
    assert postgres_service["type"] == "database"


# ============================================================================
# TESTING REQUIREMENTS VALIDATION
# ============================================================================


def test_orchestrator_testing_requirements(orchestrator_contract):
    """Test that orchestrator testing requirements are specified."""
    testing = orchestrator_contract["testing"]

    # Unit tests
    assert "unit_tests" in testing
    unit = testing["unit_tests"]
    assert "coverage_target" in unit
    assert unit["coverage_target"] == 80
    assert unit["required"] is True

    # Integration tests
    assert "integration_tests" in testing
    integration = testing["integration_tests"]
    assert integration["required"] is True
    assert "test_scenarios" in integration
    assert len(integration["test_scenarios"]) >= 4

    # Performance tests
    assert "performance_tests" in testing
    performance = testing["performance_tests"]
    assert performance["required"] is True
    assert "benchmarks" in performance


# ============================================================================
# COMPREHENSIVE CONTRACT VALIDATION
# ============================================================================


def test_orchestrator_contract_complete_validation(orchestrator_contract):
    """Comprehensive validation of orchestrator contract."""
    # Load external subcontract files
    contract = load_subcontracts(ORCHESTRATOR_CONTRACT, orchestrator_contract)
    errors = []

    # Validate schema version
    if contract.get("schema_version") != "2.0":
        errors.append("Invalid schema version (expected '2.0')")

    # Validate contract version
    if contract.get("contract_version") != "1.0.0":
        errors.append("Invalid contract version (expected '1.0.0')")

    # Validate all stages have target durations
    stages = contract["subcontracts"]["workflow_coordination"]["stages"]
    total_stage_ms = sum(stage["target_duration_ms"] for stage in stages)
    expected_total_ms = contract["performance_requirements"]["execution_time"][
        "target_ms"
    ]

    # Total target should be >= sum of stages (accounts for orchestration overhead)
    if total_stage_ms > expected_total_ms:
        errors.append(
            f"Sum of stage targets ({total_stage_ms}ms) exceeds "
            f"total target ({expected_total_ms}ms)"
        )

    # Warn if overhead seems excessive (>50% of total)
    overhead_ms = expected_total_ms - total_stage_ms
    overhead_pct = (overhead_ms / expected_total_ms) * 100
    if overhead_pct > 50:
        # This is just a warning, not an error
        print(
            f"\n⚠️ Warning: Orchestration overhead is {overhead_pct:.0f}% of total "
            f"({overhead_ms}ms / {expected_total_ms}ms)"
        )

    if errors:
        pytest.fail(
            "Contract validation failed:\n" + "\n".join(f"- {e}" for e in errors)
        )


def test_metrics_reducer_contract_complete_validation(metrics_reducer_contract):
    """Comprehensive validation of metrics reducer contract."""
    errors = []

    # Validate contract structure
    contract = metrics_reducer_contract.get("contract")
    if not contract:
        errors.append("Missing contract section")
    else:
        if "input_state" not in contract:
            errors.append("Missing input_state contract")
        if "output_state" not in contract:
            errors.append("Missing output_state contract")

    # Validate aggregation configuration
    aggregation = metrics_reducer_contract.get("aggregation")
    if aggregation:
        if aggregation.get("batch_size", 0) <= 0:
            errors.append("Invalid batch size")

    if errors:
        pytest.fail(
            "Contract validation failed:\n" + "\n".join(f"- {e}" for e in errors)
        )
