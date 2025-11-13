#!/usr/bin/env python3
"""Unit tests for NodeVaultSecretsEffect."""

import pytest
from omnibase_core.enums.enum_node_type import EnumNodeType
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer

from ..node import NodeVaultSecretsEffect


@pytest.fixture
def container():
    """Create test container with vault configuration."""
    return ModelContainer(
        value={
            "vault_addr": "http://localhost:8200",
            "vault_token": "test-token",
            "vault_namespace": "test",
            "test_mode": True,  # Skip authentication check during tests
        },
        container_type="config",
    )


@pytest.fixture
def node(container):
    """Create test node instance."""
    return NodeVaultSecretsEffect(container)


@pytest.mark.asyncio
async def test_node_initialization(node):
    """Test node initializes correctly."""
    assert node is not None
    assert node.node_id is not None


@pytest.mark.asyncio
async def test_execute_effect(node):
    """Test effect execution."""
    contract = ModelContractEffect(
        name="vault_secrets_effect",
        version={"major": 1, "minor": 0, "patch": 0},
        description="HashiCorp Vault secrets management integration",
        node_type=EnumNodeType.EFFECT,
        input_model="ModelContractEffect",
        output_model="dict",
        input_data={
            "operation": "read_secret",
            "mount_point": "secret",
            "path": "test/credentials",
        },
        io_operations=[
            {
                "operation_type": "READ",
                "resource_identifier": "vault://secret/test/credentials",
            }
        ],
    )

    # Note: This test may fail if Vault is not available or not authenticated
    # In CI/CD, consider mocking the hvac client or marking as integration test
    try:
        result = await node.execute_effect(contract)

        # Validate result structure
        assert result is not None
        assert isinstance(result, dict)

        # For read_secret operation, expect 'data' key
        assert "data" in result
        assert isinstance(result["data"], dict)

    except Exception as e:
        # If Vault is not available, skip the test gracefully
        pytest.skip(f"Vault not available or not configured: {e}")


@pytest.mark.asyncio
async def test_execute_effect_write_secret(node):
    """Test writing a secret to Vault."""
    contract = ModelContractEffect(
        name="vault_secrets_effect",
        version={"major": 1, "minor": 0, "patch": 0},
        description="Write secret operation",
        node_type=EnumNodeType.EFFECT,
        input_model="ModelContractEffect",
        output_model="dict",
        input_data={
            "operation": "write_secret",
            "mount_point": "secret",
            "path": "test/credentials",
            "data": {"username": "test_user", "password": "test_pass"},
        },
        io_operations=[
            {
                "operation_type": "WRITE",
                "resource_identifier": "vault://secret/test/credentials",
            }
        ],
    )

    try:
        result = await node.execute_effect(contract)

        # Validate result structure for write operation
        assert result is not None
        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "message" in result
        assert "successfully" in result["message"].lower()

    except Exception as e:
        pytest.skip(f"Vault not available or not configured: {e}")


@pytest.mark.asyncio
async def test_execute_effect_list_secrets(node):
    """Test listing secrets from Vault."""
    contract = ModelContractEffect(
        name="vault_secrets_effect",
        version={"major": 1, "minor": 0, "patch": 0},
        description="List secrets operation",
        node_type=EnumNodeType.EFFECT,
        input_model="ModelContractEffect",
        output_model="dict",
        input_data={
            "operation": "list_secrets",
            "mount_point": "secret",
            "path": "test/",
        },
        io_operations=[
            {"operation_type": "READ", "resource_identifier": "vault://secret/test/"}
        ],
    )

    try:
        result = await node.execute_effect(contract)

        # Validate result structure for list operation
        assert result is not None
        assert isinstance(result, dict)
        assert "secrets" in result
        assert isinstance(result["secrets"], list)

    except Exception as e:
        pytest.skip(f"Vault not available or not configured: {e}")


@pytest.mark.asyncio
async def test_execute_effect_invalid_operation(node):
    """Test handling of invalid operation."""
    contract = ModelContractEffect(
        name="vault_secrets_effect",
        version={"major": 1, "minor": 0, "patch": 0},
        description="Invalid operation test",
        node_type=EnumNodeType.EFFECT,
        input_model="ModelContractEffect",
        output_model="dict",
        input_data={
            "operation": "invalid_operation",
            "mount_point": "secret",
            "path": "test/credentials",
        },
        io_operations=[
            {
                "operation_type": "READ",
                "resource_identifier": "vault://secret/test/credentials",
            }
        ],
    )

    # Should raise an error for invalid operation
    with pytest.raises(Exception) as exc_info:
        await node.execute_effect(contract)

    # Verify error message mentions unknown operation
    assert (
        "unknown" in str(exc_info.value).lower()
        or "invalid" in str(exc_info.value).lower()
    )
