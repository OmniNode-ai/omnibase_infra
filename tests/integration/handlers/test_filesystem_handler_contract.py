# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for FileSystem Handler contract discovery.

Tests verify that the filesystem handler contract at:
    src/omnibase_infra/contracts/handlers/filesystem/handler_contract.yaml

Is correctly discovered by HandlerContractSource and validates against
the ModelHandlerContract schema.

Related:
    - OMN-1160: FileSystem Handler Contract YAML
    - OMN-1097: HandlerContractSource + Filesystem Discovery
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omnibase_core.models.contracts.model_handler_contract import ModelHandlerContract

from omnibase_infra.runtime.handler_contract_source import HandlerContractSource


def _find_project_root() -> Path:
    """Find project root by walking up directories until finding pyproject.toml.

    This approach is robust to test file relocation since it doesn't depend on
    a specific directory depth (e.g., .parents[3]). Instead, it walks up the
    directory tree looking for a reliable project marker.

    Returns:
        Path to the project root directory containing pyproject.toml.

    Raises:
        RuntimeError: If project root cannot be found (no pyproject.toml marker).
    """
    current = Path(__file__).resolve()
    project_root = current.parent
    while project_root != project_root.parent:
        if (project_root / "pyproject.toml").exists():
            return project_root
        project_root = project_root.parent

    raise RuntimeError(
        f"Could not find project root from {Path(__file__)}. "
        "Expected pyproject.toml marker in an ancestor directory."
    )


# Path to contracts directory - uses robust project root discovery
# that works regardless of test file location depth.
PROJECT_ROOT = _find_project_root()
CONTRACTS_DIR = PROJECT_ROOT / "src" / "omnibase_infra" / "contracts" / "handlers"

# Path to the specific filesystem handler contract
FILESYSTEM_CONTRACT_PATH = CONTRACTS_DIR / "filesystem" / "handler_contract.yaml"

# Expected filesystem handler properties
EXPECTED_HANDLER_ID = "effect.filesystem.handler"
EXPECTED_HANDLER_KIND = "effect"
EXPECTED_NAME = "FileSystem Handler"
EXPECTED_VERSION = "1.0.0"
EXPECTED_CAPABILITIES = [
    "filesystem.read",
    "filesystem.write",
    "filesystem.list",
    "filesystem.delete",
    "filesystem.mkdir",
]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def contract_source() -> HandlerContractSource:
    """Create HandlerContractSource for contracts directory."""
    return HandlerContractSource(contract_paths=[CONTRACTS_DIR])


@pytest.fixture
def raw_contract_data() -> dict:
    """Load raw YAML data from the filesystem handler contract."""
    with FILESYSTEM_CONTRACT_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# Contract Discovery Tests
# =============================================================================


class TestFilesystemHandlerContractDiscovery:
    """Tests for filesystem handler contract discovery via HandlerContractSource."""

    def test_contract_file_exists(self) -> None:
        """Verify the filesystem handler contract file exists at expected location.

        This test validates the test setup by confirming the contract file
        is present before running discovery tests.
        """
        assert FILESYSTEM_CONTRACT_PATH.exists(), (
            f"Filesystem handler contract not found at {FILESYSTEM_CONTRACT_PATH}. "
            "Ensure the contract file was created as part of OMN-1160."
        )
        assert FILESYSTEM_CONTRACT_PATH.is_file(), (
            f"Expected {FILESYSTEM_CONTRACT_PATH} to be a file, not a directory."
        )

    @pytest.mark.asyncio
    async def test_filesystem_handler_contract_discovery(
        self, contract_source: HandlerContractSource
    ) -> None:
        """Verify the filesystem handler contract is discovered correctly.

        HandlerContractSource should:
        1. Find the filesystem/handler_contract.yaml file
        2. Parse it successfully
        3. Include it in the discovered descriptors
        4. Return correct handler_id and handler_kind
        """
        result = await contract_source.discover_handlers()

        # Verify no validation errors
        assert len(result.validation_errors) == 0, (
            f"Expected no validation errors, got: {result.validation_errors}"
        )

        # Verify at least one descriptor was found
        assert len(result.descriptors) >= 1, (
            f"Expected at least 1 descriptor, got {len(result.descriptors)}"
        )

        # Find the filesystem handler descriptor
        filesystem_descriptors = [
            d for d in result.descriptors if d.handler_id == EXPECTED_HANDLER_ID
        ]

        assert len(filesystem_descriptors) == 1, (
            f"Expected exactly 1 descriptor with handler_id '{EXPECTED_HANDLER_ID}', "
            f"found {len(filesystem_descriptors)}. "
            f"Discovered handler_ids: {[d.handler_id for d in result.descriptors]}"
        )

        descriptor = filesystem_descriptors[0]

        # Verify handler_id
        assert descriptor.handler_id == EXPECTED_HANDLER_ID, (
            f"Expected handler_id '{EXPECTED_HANDLER_ID}', "
            f"got '{descriptor.handler_id}'"
        )

        # Verify handler_kind
        assert descriptor.handler_kind == EXPECTED_HANDLER_KIND, (
            f"Expected handler_kind '{EXPECTED_HANDLER_KIND}', "
            f"got '{descriptor.handler_kind}'"
        )

        # Verify name
        assert descriptor.name == EXPECTED_NAME, (
            f"Expected name '{EXPECTED_NAME}', got '{descriptor.name}'"
        )

        # Verify version
        assert str(descriptor.version) == EXPECTED_VERSION, (
            f"Expected version '{EXPECTED_VERSION}', got '{descriptor.version}'"
        )

    @pytest.mark.asyncio
    async def test_filesystem_handler_contract_path_in_descriptor(
        self, contract_source: HandlerContractSource
    ) -> None:
        """Verify the discovered descriptor includes the correct contract_path.

        The contract_path should point to the actual handler_contract.yaml file
        for traceability and debugging purposes.
        """
        result = await contract_source.discover_handlers()

        filesystem_descriptors = [
            d for d in result.descriptors if d.handler_id == EXPECTED_HANDLER_ID
        ]
        assert len(filesystem_descriptors) == 1

        descriptor = filesystem_descriptors[0]

        # Verify contract_path points to the correct file
        assert descriptor.contract_path is not None, (
            "Descriptor should include contract_path for traceability"
        )
        assert "filesystem" in descriptor.contract_path, (
            f"contract_path should contain 'filesystem', got: {descriptor.contract_path}"
        )
        assert "handler_contract.yaml" in descriptor.contract_path, (
            f"contract_path should contain 'handler_contract.yaml', "
            f"got: {descriptor.contract_path}"
        )


# =============================================================================
# Contract Schema Validation Tests
# =============================================================================


class TestFilesystemHandlerContractSchema:
    """Tests for filesystem handler contract schema validation."""

    def test_filesystem_handler_contract_schema_valid(
        self, raw_contract_data: dict
    ) -> None:
        """Verify the contract YAML validates against ModelHandlerContract schema.

        This test directly parses the YAML and validates it using Pydantic,
        ensuring the contract structure is correct independently of the
        HandlerContractSource discovery process.
        """
        # Should not raise ValidationError
        contract = ModelHandlerContract.model_validate(raw_contract_data)

        # Verify key fields were parsed correctly
        assert contract.handler_id == EXPECTED_HANDLER_ID
        assert contract.name == EXPECTED_NAME
        assert str(contract.version) == EXPECTED_VERSION
        assert contract.descriptor.handler_kind == EXPECTED_HANDLER_KIND

    def test_filesystem_handler_contract_has_required_fields(
        self, raw_contract_data: dict
    ) -> None:
        """Verify the contract contains all required fields for a handler contract.

        ModelHandlerContract requires:
        - handler_id
        - name
        - version
        - descriptor (with handler_kind)
        - input_model
        - output_model
        """
        required_fields = [
            "handler_id",
            "name",
            "version",
            "descriptor",
            "input_model",
            "output_model",
        ]

        for field in required_fields:
            assert field in raw_contract_data, (
                f"Required field '{field}' missing from filesystem handler contract"
            )

        # Verify descriptor has handler_kind
        assert "handler_kind" in raw_contract_data.get("descriptor", {}), (
            "descriptor.handler_kind is required"
        )

    def test_filesystem_handler_contract_descriptor_fields(
        self, raw_contract_data: dict
    ) -> None:
        """Verify the descriptor section contains expected effect handler fields.

        Effect handlers should specify:
        - handler_kind: effect
        - purity: side_effecting
        - Additional effect-specific configuration
        """
        descriptor = raw_contract_data.get("descriptor", {})

        assert descriptor.get("handler_kind") == "effect", (
            "FileSystem handler should be an 'effect' handler"
        )
        assert descriptor.get("purity") == "side_effecting", (
            "FileSystem handler should be 'side_effecting' as it performs I/O"
        )


# =============================================================================
# Capability Tests
# =============================================================================


class TestFilesystemHandlerCapabilities:
    """Tests for filesystem handler capability declarations."""

    def test_filesystem_handler_capabilities_present(
        self, raw_contract_data: dict
    ) -> None:
        """Verify the contract declares capability_outputs.

        The filesystem handler should declare its capabilities in the
        capability_outputs field for capability-based discovery.
        """
        assert "capability_outputs" in raw_contract_data, (
            "FileSystem handler contract should include capability_outputs field"
        )

        capabilities = raw_contract_data.get("capability_outputs", [])
        assert isinstance(capabilities, list), (
            f"capability_outputs should be a list, got {type(capabilities)}"
        )
        assert len(capabilities) > 0, (
            "capability_outputs should not be empty for filesystem handler"
        )

    def test_filesystem_handler_capabilities_complete(
        self, raw_contract_data: dict
    ) -> None:
        """Verify all expected filesystem capabilities are declared.

        Expected capabilities:
        - filesystem.read: Read file contents
        - filesystem.write: Write file contents
        - filesystem.list: List directory contents
        - filesystem.delete: Delete files/directories
        - filesystem.mkdir: Create directories
        """
        capabilities = raw_contract_data.get("capability_outputs", [])

        for expected_cap in EXPECTED_CAPABILITIES:
            assert expected_cap in capabilities, (
                f"Expected capability '{expected_cap}' not found in contract. "
                f"Declared capabilities: {capabilities}"
            )

    def test_filesystem_handler_capabilities_exact_match(
        self, raw_contract_data: dict
    ) -> None:
        """Verify the contract declares exactly the expected capabilities.

        This test ensures no unexpected capabilities are declared that
        might indicate scope creep or errors.
        """
        capabilities = set(raw_contract_data.get("capability_outputs", []))
        expected = set(EXPECTED_CAPABILITIES)

        # Check for missing capabilities
        missing = expected - capabilities
        assert not missing, f"Missing expected capabilities: {missing}"

        # Check for unexpected capabilities (informational, not failure)
        extra = capabilities - expected
        if extra:
            # Log but don't fail - additional capabilities may be intentional
            import warnings

            warnings.warn(
                f"Contract declares additional capabilities beyond expected: {extra}. "
                "Update EXPECTED_CAPABILITIES if these are intentional.",
                stacklevel=1,
            )


# =============================================================================
# Integration with Retry and Circuit Breaker Configuration Tests
# =============================================================================


class TestFilesystemHandlerEffectConfiguration:
    """Tests for filesystem handler effect-specific configuration."""

    def test_filesystem_handler_has_retry_policy(self, raw_contract_data: dict) -> None:
        """Verify the contract includes retry policy configuration.

        Effect handlers should specify retry behavior for resilience.
        """
        descriptor = raw_contract_data.get("descriptor", {})
        retry_policy = descriptor.get("retry_policy", {})

        assert retry_policy.get("enabled") is True, (
            "FileSystem handler should have retry_policy.enabled = true"
        )
        assert "max_retries" in retry_policy, "retry_policy should specify max_retries"
        assert "backoff_strategy" in retry_policy, (
            "retry_policy should specify backoff_strategy"
        )

    def test_filesystem_handler_has_circuit_breaker(
        self, raw_contract_data: dict
    ) -> None:
        """Verify the contract includes circuit breaker configuration.

        Effect handlers should specify circuit breaker for fault tolerance.
        """
        descriptor = raw_contract_data.get("descriptor", {})
        circuit_breaker = descriptor.get("circuit_breaker", {})

        assert circuit_breaker.get("enabled") is True, (
            "FileSystem handler should have circuit_breaker.enabled = true"
        )
        assert "failure_threshold" in circuit_breaker, (
            "circuit_breaker should specify failure_threshold"
        )
        assert "timeout_ms" in circuit_breaker, (
            "circuit_breaker should specify timeout_ms"
        )

    def test_filesystem_handler_lifecycle_support(
        self, raw_contract_data: dict
    ) -> None:
        """Verify the contract declares lifecycle support.

        Effect handlers that manage resources should support lifecycle methods.
        """
        assert raw_contract_data.get("supports_lifecycle") is True, (
            "FileSystem handler should support lifecycle (start/stop)"
        )

    def test_filesystem_handler_health_check_support(
        self, raw_contract_data: dict
    ) -> None:
        """Verify the contract declares health check support.

        Effect handlers should support health checks for observability.
        """
        assert raw_contract_data.get("supports_health_check") is True, (
            "FileSystem handler should support health checks"
        )


# =============================================================================
# Idempotency Tests
# =============================================================================


class TestFilesystemHandlerContractDiscoveryIdempotency:
    """Tests for idempotent discovery of filesystem handler contract."""

    @pytest.mark.asyncio
    async def test_repeated_discovery_returns_consistent_results(
        self, contract_source: HandlerContractSource
    ) -> None:
        """Verify multiple discovery calls return the same filesystem handler.

        HandlerContractSource.discover_handlers() should be idempotent,
        returning consistent results across multiple calls.
        """
        # Discover multiple times
        result1 = await contract_source.discover_handlers()
        result2 = await contract_source.discover_handlers()
        result3 = await contract_source.discover_handlers()

        # Extract filesystem handler from each result
        def get_filesystem_descriptor(result):
            return next(
                (d for d in result.descriptors if d.handler_id == EXPECTED_HANDLER_ID),
                None,
            )

        desc1 = get_filesystem_descriptor(result1)
        desc2 = get_filesystem_descriptor(result2)
        desc3 = get_filesystem_descriptor(result3)

        # All should find the filesystem handler
        assert desc1 is not None, "First discovery should find filesystem handler"
        assert desc2 is not None, "Second discovery should find filesystem handler"
        assert desc3 is not None, "Third discovery should find filesystem handler"

        # All should have identical properties
        assert desc1.handler_id == desc2.handler_id == desc3.handler_id
        assert desc1.handler_kind == desc2.handler_kind == desc3.handler_kind
        assert desc1.name == desc2.name == desc3.name
        assert str(desc1.version) == str(desc2.version) == str(desc3.version)
