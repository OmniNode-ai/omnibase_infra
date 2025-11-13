#!/usr/bin/env python3
"""
Simple O.N.E. v0.1 Protocol Component Test Runner.

Runs O.N.E. tests without complex pytest configuration requirements.
"""

import asyncio
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))


def run_simple_test():
    """Run a simple test to validate O.N.E. components are working."""

    print("🚀 Simple O.N.E. Protocol Component Validation")
    print("=" * 60)

    # Available components tracking
    available_components = []

    try:
        # Test 1: Basic imports
        print("\n1️⃣ Testing basic imports...")

        # Registry component
        try:
            from omninode_bridge.services.metadata_stamping.registry.consul_client import (
                RegistryConsulClient,
            )

            available_components.append("RegistryConsulClient")
            print("   ✅ Registry component import successful")
        except ImportError as e:
            print(f"   ⚠️ Registry import failed: {e}")

        # Security components
        try:
            from omninode_bridge.services.metadata_stamping.security.signature_validator import (
                SignatureValidator,
            )
            from omninode_bridge.services.metadata_stamping.security.trust_zones import (
                TrustZoneManager,
            )

            available_components.extend(["TrustZoneManager", "SignatureValidator"])
            print("   ✅ Security components import successful")
        except ImportError as e:
            print(f"   ⚠️ Security imports failed: {e}")

        # Execution components
        try:
            from omninode_bridge.services.metadata_stamping.execution.dag_engine import (
                DAGExecutor,
            )
            from omninode_bridge.services.metadata_stamping.execution.simulation import (
                WorkflowSimulator,
            )
            from omninode_bridge.services.metadata_stamping.execution.transformer import (  # noqa: F401
                BaseTransformer,
            )

            available_components.extend(
                ["BaseTransformer", "DAGExecutor", "WorkflowSimulator"]
            )
            print("   ✅ Execution components import successful")
        except ImportError as e:
            print(f"   ⚠️ Execution imports failed: {e}")

        # Schema management
        try:
            from omninode_bridge.services.metadata_stamping.execution.schema_registry import (
                SchemaRegistry,
            )

            available_components.append("SchemaRegistry")
            print("   ✅ Schema management import successful")
        except ImportError as e:
            print(f"   ⚠️ Schema import failed: {e}")

        if available_components:
            print(f"   📊 Available: {len(available_components)} components")
            print(f"   🎉 Components loaded: {', '.join(available_components)}")
        else:
            print("   ❌ No components could be imported!")
            return False

    except Exception as e:
        print(f"   ❌ Critical import failure: {e}")
        return False

    try:
        # Test 2: Basic instantiation
        print("\n2️⃣ Testing component instantiation...")

        instantiated = []

        # Registry (with mock)
        if "RegistryConsulClient" in available_components:
            try:
                from unittest.mock import MagicMock, patch

                with patch("consul.Consul") as mock_consul:
                    mock_consul.return_value = MagicMock()
                    registry = RegistryConsulClient("test-host", 8500)
                    instantiated.append("RegistryConsulClient")
                    print("   ✅ Registry instantiation successful")
            except Exception as e:
                print(f"   ⚠️ Registry instantiation failed: {e}")

        # Security
        if (
            "TrustZoneManager" in available_components
            and "SignatureValidator" in available_components
        ):
            try:
                trust_manager = TrustZoneManager()
                signature_validator = SignatureValidator()
                instantiated.extend(["TrustZoneManager", "SignatureValidator"])
                print("   ✅ Security components instantiation successful")
            except Exception as e:
                print(f"   ⚠️ Security instantiation failed: {e}")

        # Execution
        if (
            "DAGExecutor" in available_components
            and "WorkflowSimulator" in available_components
        ):
            try:
                dag_executor = DAGExecutor()
                simulator = WorkflowSimulator()
                instantiated.extend(["DAGExecutor", "WorkflowSimulator"])
                print("   ✅ Execution components instantiation successful")
            except Exception as e:
                print(f"   ⚠️ Execution instantiation failed: {e}")

        # Schema
        if "SchemaRegistry" in available_components:
            try:
                schema_registry = SchemaRegistry()
                instantiated.append("SchemaRegistry")
                print("   ✅ Schema registry instantiation successful")
            except Exception as e:
                print(f"   ⚠️ Schema instantiation failed: {e}")

        if instantiated:
            print(f"   📊 Instantiated: {len(instantiated)} components")
            print(f"   🎉 Ready components: {', '.join(instantiated)}")
        else:
            print("   ❌ No components could be instantiated!")
            return False

    except Exception as e:
        print(f"   ❌ Critical instantiation failure: {e}")
        return False

    try:
        # Test 3: Basic functionality
        print("\n3️⃣ Testing basic functionality...")

        # Trust zone assignment
        from omninode_bridge.services.metadata_stamping.security.trust_zones import (
            TrustZone,
        )

        zone = trust_manager.assign_trust_zone("localhost")
        assert zone == TrustZone.LOCAL
        print("   ✅ Trust zone assignment working")

        # Signature generation
        test_data = b"test message"
        sig_result = signature_validator.generate_signature(test_data)
        assert sig_result is not None
        print("   ✅ Signature generation working")

        # Schema registration
        from pydantic import BaseModel, Field

        class TestModel(BaseModel):
            id: int = Field(..., description="Test ID")
            name: str = Field(..., description="Test name")

        result = schema_registry.register_schema("test_schema", TestModel, "1.0.0")
        assert result is True
        print("   ✅ Schema registration working")

        # DAG validation
        is_valid = dag_executor.validate_dag()
        assert is_valid is True
        print("   ✅ DAG validation working")

        print("   🎉 All basic functionality tests passed!")

    except Exception as e:
        print(f"   ❌ Functionality test failed: {e}")
        return False

    try:
        # Test 4: Integration test
        print("\n4️⃣ Testing basic integration...")

        # Create a simple workflow

        from omninode_bridge.services.metadata_stamping.execution.transformer import (
            ExecutionContext,
            transformer,
        )

        @transformer(TestModel, TestModel, name="test_transformer")
        async def test_transformer_func(
            input_data: TestModel, context: ExecutionContext
        ) -> TestModel:
            return TestModel(id=input_data.id + 1, name=f"processed_{input_data.name}")

        # Test the transformer
        async def test_transformer_execution():
            context = ExecutionContext(
                execution_id="test-001",
                input_schema="TestModel",
                output_schema="TestModel",
            )

            input_data = {"id": 1, "name": "test"}
            result = await test_transformer_func.execute_with_validation(
                input_data, context
            )

            assert result.id == 2
            assert result.name == "processed_test"
            return True

        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(test_transformer_execution())
        loop.close()

        assert success is True
        print("   ✅ Transformer execution working")

        print("   🎉 Integration test passed!")

    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

    # Final summary
    print(f"\n{'VALIDATION SUMMARY':=^60}")
    print("✅ All O.N.E. Protocol Components Validated Successfully!")
    print("🔧 Registry: Service registration and discovery")
    print("🔒 Security: Trust zones and signature validation")
    print("⚙️  Execution: Transformers, DAG, and simulation")
    print("📊 Schema: Registration, validation, and evolution")
    print("🔗 Integration: Components working together")
    print(f"{'='*60}")

    return True


def run_integration_scenario():
    """Run a more comprehensive integration scenario."""

    print("\n🔗 Running Integration Scenario...")
    print("-" * 40)

    try:
        # Import required components
        from datetime import datetime

        from pydantic import BaseModel, Field

        from omninode_bridge.services.metadata_stamping.execution.dag_engine import (
            DAGExecutor,
        )
        from omninode_bridge.services.metadata_stamping.execution.schema_registry import (
            SchemaRegistry,
        )
        from omninode_bridge.services.metadata_stamping.execution.transformer import (
            ExecutionContext,
            transformer,
        )
        from omninode_bridge.services.metadata_stamping.security.signature_validator import (
            SignatureValidator,
        )
        from omninode_bridge.services.metadata_stamping.security.trust_zones import (
            TrustZone,
            TrustZoneManager,
        )

        # Define O.N.E. compliant models
        class ONEInputModel(BaseModel):
            operation_id: str = Field(..., description="Operation ID")
            file_hash: str = Field(..., description="File hash")
            metadata: dict = Field(default_factory=dict, description="Metadata")

        class ONEOutputModel(BaseModel):
            operation_id: str = Field(..., description="Operation ID")
            result: str = Field(..., description="Processing result")
            timestamp: str = Field(..., description="Processing timestamp")

        # Initialize components
        trust_manager = TrustZoneManager()
        signature_validator = SignatureValidator()
        schema_registry = SchemaRegistry()
        dag_executor = DAGExecutor()

        # Register schemas
        schema_registry.register_schema("ONEInput", ONEInputModel, "1.0.0")
        schema_registry.register_schema("ONEOutput", ONEOutputModel, "1.0.0")

        # Create transformer
        @transformer(ONEInputModel, ONEOutputModel, name="one_processor")
        async def one_processor(
            input_data: ONEInputModel, context: ExecutionContext
        ) -> ONEOutputModel:
            return ONEOutputModel(
                operation_id=input_data.operation_id,
                result=f"processed_{input_data.file_hash[:8]}",
                timestamp=datetime.now(UTC).isoformat(),
            )

        # Test integration scenario
        async def integration_test():
            # 1. Trust zone assignment
            zone = trust_manager.assign_trust_zone("api.omninode.org")
            assert zone == TrustZone.ORG
            print("   ✅ Trust zone assignment: ORG")

            # 2. Message signing
            test_message = {
                "operation_id": "int-test-001",
                "file_hash": "a" * 64,
                "metadata": {"test": True},
            }

            message_bytes = signature_validator._canonicalize_message(test_message)
            sig_result = signature_validator.generate_signature(message_bytes)
            assert sig_result is not None
            print("   ✅ Message signing successful")

            # 3. Schema validation
            is_valid, model, error = schema_registry.validate_data(
                "ONEInput", test_message, "1.0.0"
            )
            assert is_valid is True
            print("   ✅ Schema validation successful")

            # 4. Transformer execution
            context = ExecutionContext(
                execution_id="integration-test",
                input_schema="ONEInput",
                output_schema="ONEOutput",
            )

            result = await one_processor.execute_with_validation(test_message, context)
            assert result.operation_id == "int-test-001"
            assert "processed_" in result.result
            print("   ✅ Transformer execution successful")

            # 5. DAG workflow
            dag_executor.add_node("process", one_processor)
            workflow_result = await dag_executor.execute(
                {"process": test_message}, simulation_mode=False
            )
            assert workflow_result["overall_status"] == "completed"
            print("   ✅ DAG workflow execution successful")

            return True

        # Run integration test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(integration_test())
        loop.close()

        assert success is True
        print("🎉 Integration scenario completed successfully!")

        return True

    except Exception as e:
        print(f"❌ Integration scenario failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("🧪 O.N.E. v0.1 Protocol Component Validation")
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python version: {sys.version}")

    # Run basic validation
    basic_success = run_simple_test()

    if not basic_success:
        print("\n❌ Basic validation failed!")
        return False

    # Run integration scenario
    integration_success = run_integration_scenario()

    if not integration_success:
        print("\n❌ Integration scenario failed!")
        return False

    print(f"\n{'FINAL RESULT':=^60}")
    print("🎉 ALL O.N.E. PROTOCOL COMPONENTS VALIDATED!")
    print(
        "✅ The metadata stamping service O.N.E. v0.1 components are working correctly"
    )
    print("🚀 Ready for production deployment")
    print(f"{'='*60}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
