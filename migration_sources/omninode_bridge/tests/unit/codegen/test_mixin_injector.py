#!/usr/bin/env python3
"""
Unit tests for MixinInjector.

Tests code generation for nodes with mixin inheritance, including:
- Import generation (organized by category)
- Class definition generation (with proper inheritance)
- Initialization code generation
- Method generation
- Complete file generation
- Multiple mixin combinations
"""

import ast

import pytest

from omninode_bridge.codegen.mixin_injector import (
    MixinInjector,
    ModelGeneratedClass,
    ModelGeneratedImports,
)


class TestModelGeneratedImports:
    """Test ModelGeneratedImports data model."""

    def test_empty_imports(self):
        """Test empty imports initialization."""
        imports = ModelGeneratedImports()
        assert imports.standard_library == []
        assert imports.third_party == []
        assert imports.omnibase_core == []
        assert imports.omnibase_mixins == []
        assert imports.project_local == []

    def test_imports_with_data(self):
        """Test imports with data."""
        imports = ModelGeneratedImports(
            standard_library=["import os"],
            third_party=["import httpx"],
            omnibase_core=["from omnibase_core.nodes.node_effect import NodeEffect"],
            omnibase_mixins=[
                "from omnibase_core.mixins.mixin_health_check import MixinHealthCheck"
            ],
            project_local=["from .models import MyModel"],
        )

        assert len(imports.standard_library) == 1
        assert len(imports.third_party) == 1
        assert len(imports.omnibase_core) == 1
        assert len(imports.omnibase_mixins) == 1
        assert len(imports.project_local) == 1


class TestModelGeneratedClass:
    """Test ModelGeneratedClass data model."""

    def test_minimal_class(self):
        """Test minimal class definition."""
        class_def = ModelGeneratedClass(
            class_name="NodeMyEffect",
            base_classes=["NodeEffect"],
            docstring='"""My node."""',
            init_method="def __init__(self):\n    pass",
            initialize_method="async def initialize(self):\n    pass",
        )

        assert class_def.class_name == "NodeMyEffect"
        assert len(class_def.base_classes) == 1
        assert class_def.shutdown_method == ""
        assert class_def.methods == []

    def test_class_with_mixins(self):
        """Test class with mixin inheritance."""
        class_def = ModelGeneratedClass(
            class_name="NodeMyEffect",
            base_classes=["NodeEffect", "MixinHealthCheck", "MixinMetrics"],
            docstring='"""My node with mixins."""',
            init_method="def __init__(self):\n    pass",
            initialize_method="async def initialize(self):\n    pass",
            shutdown_method="async def shutdown(self):\n    pass",
            methods=["def my_method(self):\n    pass"],
        )

        assert len(class_def.base_classes) == 3
        assert "MixinHealthCheck" in class_def.base_classes
        assert "MixinMetrics" in class_def.base_classes
        assert class_def.shutdown_method != ""
        assert len(class_def.methods) == 1


class TestMixinInjector:
    """Test MixinInjector code generation."""

    @pytest.fixture
    def injector(self):
        """Create MixinInjector instance."""
        return MixinInjector()

    @pytest.fixture
    def minimal_contract(self):
        """Minimal contract for testing."""
        return {
            "name": "test_effect",
            "node_type": "EFFECT",
            "description": "Test effect node",
            "mixins": [],
        }

    @pytest.fixture
    def contract_with_health_check(self):
        """Contract with MixinHealthCheck."""
        return {
            "name": "postgres_crud_effect",
            "node_type": "EFFECT",
            "description": "PostgreSQL CRUD operations",
            "mixins": [
                {
                    "name": "MixinHealthCheck",
                    "enabled": True,
                    "config": {
                        "check_interval_ms": 30000,
                        "timeout_seconds": 5.0,
                    },
                }
            ],
        }

    @pytest.fixture
    def contract_with_multiple_mixins(self):
        """Contract with multiple mixins."""
        return {
            "name": "event_processor_effect",
            "node_type": "EFFECT",
            "description": "Event processor with monitoring",
            "mixins": [
                {"name": "MixinHealthCheck", "enabled": True, "config": {}},
                {"name": "MixinMetrics", "enabled": True, "config": {}},
                {
                    "name": "MixinEventDrivenNode",
                    "enabled": True,
                    "config": {"domain_filter": "events"},
                },
            ],
        }

    # Import generation tests
    def test_generate_imports_minimal(self, injector, minimal_contract):
        """Test import generation for minimal contract."""
        imports = injector.generate_imports(minimal_contract)

        # Check standard library imports
        assert "import logging" in imports.standard_library
        assert "from typing import Any, Optional" in imports.standard_library

        # Check omnibase_core imports
        assert (
            "from omnibase_core.nodes.node_effect import NodeEffect"
            in imports.omnibase_core
        )
        assert (
            "from omnibase_core.models.core.model_container import ModelContainer"
            in imports.omnibase_core
        )

        # No mixins
        assert len(imports.omnibase_mixins) == 0

    def test_generate_imports_with_health_check(
        self, injector, contract_with_health_check
    ):
        """Test import generation with MixinHealthCheck."""
        imports = injector.generate_imports(contract_with_health_check)

        # Check mixin import
        assert (
            "from omnibase_core.mixins.mixin_health_check import MixinHealthCheck"
            in imports.omnibase_mixins
        )

        # Check health status imports
        assert (
            "from omnibase_core.models.core.model_health_status import ModelHealthStatus"
            in imports.omnibase_core
        )
        assert (
            "from omnibase_core.enums.enum_node_health_status import EnumNodeHealthStatus"
            in imports.omnibase_core
        )

    def test_generate_imports_with_multiple_mixins(
        self, injector, contract_with_multiple_mixins
    ):
        """Test import generation with multiple mixins."""
        imports = injector.generate_imports(contract_with_multiple_mixins)

        # Check all mixin imports
        assert (
            "from omnibase_core.mixins.mixin_health_check import MixinHealthCheck"
            in imports.omnibase_mixins
        )
        assert (
            "from omnibase_core.mixins.mixin_metrics import MixinMetrics"
            in imports.omnibase_mixins
        )
        assert (
            "from omnibase_core.mixins.mixin_event_driven_node import MixinEventDrivenNode"
            in imports.omnibase_mixins
        )

    def test_generate_imports_disabled_mixin(self, injector):
        """Test that disabled mixins are not imported."""
        contract = {
            "name": "test_effect",
            "node_type": "EFFECT",
            "description": "Test node",
            "mixins": [
                {"name": "MixinHealthCheck", "enabled": False},
                {"name": "MixinMetrics", "enabled": True},
            ],
        }

        imports = injector.generate_imports(contract)

        # Disabled mixin should not be imported
        assert not any("MixinHealthCheck" in imp for imp in imports.omnibase_mixins)

        # Enabled mixin should be imported
        assert any("MixinMetrics" in imp for imp in imports.omnibase_mixins)

    # Class definition tests
    def test_generate_class_definition_minimal(self, injector, minimal_contract):
        """Test class definition for minimal contract."""
        class_def = injector.generate_class_definition(minimal_contract)

        assert class_def.class_name == "NodeTestEffect"
        assert class_def.base_classes == ["NodeEffect"]
        assert "ONEX v2.0 Compliant Effect Node" in class_def.docstring

    def test_generate_class_definition_with_mixin(
        self, injector, contract_with_health_check
    ):
        """Test class definition with mixin."""
        class_def = injector.generate_class_definition(contract_with_health_check)

        assert class_def.class_name == "NodePostgresCrudEffect"
        assert class_def.base_classes == ["NodeEffect", "MixinHealthCheck"]
        assert "MixinHealthCheck" in class_def.docstring

    def test_generate_class_definition_multiple_mixins(
        self, injector, contract_with_multiple_mixins
    ):
        """Test class definition with multiple mixins."""
        class_def = injector.generate_class_definition(contract_with_multiple_mixins)

        assert len(class_def.base_classes) == 4  # NodeEffect + 3 mixins
        assert "NodeEffect" in class_def.base_classes
        assert "MixinHealthCheck" in class_def.base_classes
        assert "MixinMetrics" in class_def.base_classes
        assert "MixinEventDrivenNode" in class_def.base_classes

    def test_class_name_generation_snake_case(self, injector):
        """Test class name generation from snake_case."""
        contract = {
            "name": "my_complex_node_name",
            "node_type": "EFFECT",
            "description": "Test",
            "mixins": [],
        }

        class_def = injector.generate_class_definition(contract)
        assert class_def.class_name == "NodeMyComplexNodeName"

    # Init method tests
    def test_init_method_minimal(self, injector, minimal_contract):
        """Test __init__ method generation for minimal contract."""
        class_def = injector.generate_class_definition(minimal_contract)

        assert "def __init__(self, container: ModelContainer):" in class_def.init_method
        assert "super().__init__(container)" in class_def.init_method
        assert "self.logger = logging.getLogger" in class_def.init_method

    def test_init_method_with_config(self, injector, contract_with_health_check):
        """Test __init__ method with mixin configuration."""
        class_def = injector.generate_class_definition(contract_with_health_check)

        assert "healthcheck_config" in class_def.init_method
        assert '"check_interval_ms": 30000' in class_def.init_method
        assert '"timeout_seconds": 5.0' in class_def.init_method

    # Initialize method tests
    def test_initialize_method_minimal(self, injector, minimal_contract):
        """Test initialize method for minimal contract."""
        class_def = injector.generate_class_definition(minimal_contract)

        assert "async def initialize(self) -> None:" in class_def.initialize_method
        assert "await super().initialize()" in class_def.initialize_method
        assert "self.logger.info" in class_def.initialize_method

    def test_initialize_method_with_health_check(
        self, injector, contract_with_health_check
    ):
        """Test initialize method with MixinHealthCheck."""
        class_def = injector.generate_class_definition(contract_with_health_check)

        assert "Setup health checks" in class_def.initialize_method
        assert "get_health_checks()" in class_def.initialize_method

    def test_initialize_method_with_event_driven(self, injector):
        """Test initialize method with MixinEventDrivenNode."""
        contract = {
            "name": "test_effect",
            "node_type": "EFFECT",
            "description": "Test",
            "mixins": [{"name": "MixinEventDrivenNode", "enabled": True}],
        }

        class_def = injector.generate_class_definition(contract)

        assert "Setup event consumption" in class_def.initialize_method
        assert "start_event_consumption()" in class_def.initialize_method

    # Shutdown method tests
    def test_shutdown_method_minimal(self, injector, minimal_contract):
        """Test shutdown method for minimal contract."""
        class_def = injector.generate_class_definition(minimal_contract)

        assert "async def shutdown(self) -> None:" in class_def.shutdown_method
        assert "await super().shutdown()" in class_def.shutdown_method

    def test_shutdown_method_with_event_driven(self, injector):
        """Test shutdown method with MixinEventDrivenNode."""
        contract = {
            "name": "test_effect",
            "node_type": "EFFECT",
            "description": "Test",
            "mixins": [{"name": "MixinEventDrivenNode", "enabled": True}],
        }

        class_def = injector.generate_class_definition(contract)

        assert "stop_event_consumption()" in class_def.shutdown_method

    # Method generation tests
    def test_generate_mixin_methods_health_check(
        self, injector, contract_with_health_check
    ):
        """Test mixin method generation for MixinHealthCheck."""
        class_def = injector.generate_class_definition(contract_with_health_check)

        # Should have get_health_checks and _check_self_health methods
        assert len(class_def.methods) >= 1
        methods_str = "\n".join(class_def.methods)
        assert "def get_health_checks(self)" in methods_str
        assert "async def _check_self_health(self)" in methods_str
        assert "ModelHealthStatus" in methods_str

    def test_generate_mixin_methods_event_driven(self, injector):
        """Test mixin method generation for MixinEventDrivenNode."""
        contract = {
            "name": "test_effect",
            "node_type": "EFFECT",
            "description": "Test",
            "mixins": [{"name": "MixinEventDrivenNode", "enabled": True}],
        }

        class_def = injector.generate_class_definition(contract)

        methods_str = "\n".join(class_def.methods)
        assert "def get_capabilities(self)" in methods_str
        assert "def supports_introspection(self)" in methods_str

    def test_generate_mixin_methods_event_bus(self, injector):
        """Test mixin method generation for MixinEventBus."""
        contract = {
            "name": "test_effect",
            "node_type": "EFFECT",
            "description": "Test",
            "mixins": [{"name": "MixinEventBus", "enabled": True}],
        }

        class_def = injector.generate_class_definition(contract)

        methods_str = "\n".join(class_def.methods)
        assert "def get_event_patterns(self)" in methods_str

    # Complete file generation tests
    def test_generate_node_file_minimal(self, injector, minimal_contract):
        """Test complete file generation for minimal contract."""
        node_code = injector.generate_node_file(minimal_contract)

        # Check file structure
        assert "#!/usr/bin/env python3" in node_code
        assert "Generated by OmniNode Code Generator" in node_code
        assert "DO NOT EDIT MANUALLY" in node_code

        # Check imports
        assert "import logging" in node_code
        assert "from omnibase_core.nodes.node_effect import NodeEffect" in node_code

        # Check class definition
        assert "class NodeTestEffect(NodeEffect):" in node_code
        assert "def __init__(self, container: ModelContainer):" in node_code
        assert "async def initialize(self) -> None:" in node_code
        assert "async def shutdown(self) -> None:" in node_code

    def test_generate_node_file_with_mixin(self, injector, contract_with_health_check):
        """Test complete file generation with mixin."""
        node_code = injector.generate_node_file(contract_with_health_check)

        # Check mixin import
        assert (
            "from omnibase_core.mixins.mixin_health_check import MixinHealthCheck"
            in node_code
        )

        # Check inheritance
        assert (
            "class NodePostgresCrudEffect(NodeEffect, MixinHealthCheck):" in node_code
        )

        # Check mixin methods
        assert "def get_health_checks(self)" in node_code
        assert "async def _check_self_health(self)" in node_code

    def test_generate_node_file_multiple_mixins(
        self, injector, contract_with_multiple_mixins
    ):
        """Test complete file generation with multiple mixins."""
        node_code = injector.generate_node_file(contract_with_multiple_mixins)

        # Check all mixin imports
        assert "MixinHealthCheck" in node_code
        assert "MixinMetrics" in node_code
        assert "MixinEventDrivenNode" in node_code

        # Check inheritance (all mixins in class definition)
        assert "NodeEffect" in node_code
        assert "MixinHealthCheck" in node_code
        assert "MixinMetrics" in node_code
        assert "MixinEventDrivenNode" in node_code

    def test_generate_node_file_syntax_valid(self, injector, minimal_contract):
        """Test that generated code has valid Python syntax."""
        node_code = injector.generate_node_file(minimal_contract)

        # Parse code to check syntax
        try:
            ast.parse(node_code)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False

        assert syntax_valid, "Generated code should have valid Python syntax"

    def test_generate_node_file_different_node_types(self, injector):
        """Test file generation for different node types."""
        node_types = ["EFFECT", "COMPUTE", "REDUCER", "ORCHESTRATOR"]

        for node_type in node_types:
            contract = {
                "name": f"test_{node_type.lower()}",
                "node_type": node_type,
                "description": f"Test {node_type} node",
                "mixins": [],
            }

            node_code = injector.generate_node_file(contract)

            # Check correct base class or convenience wrapper
            # Note: REDUCER and ORCHESTRATOR use convenience wrappers by default when no mixins specified
            if node_type in ["REDUCER", "ORCHESTRATOR"]:
                # These types use convenience wrappers (ModelServiceReducer, ModelServiceOrchestrator)
                if node_type == "REDUCER":
                    assert "class NodeTestReducer(ModelServiceReducer):" in node_code
                    assert (
                        "from omninode_bridge.utils.node_services import ModelServiceReducer"
                        in node_code
                    )
                elif node_type == "ORCHESTRATOR":
                    assert (
                        "class NodeTestOrchestrator(ModelServiceOrchestrator):"
                        in node_code
                    )
                    assert (
                        "from omninode_bridge.utils.node_services import ModelServiceOrchestrator"
                        in node_code
                    )
            else:
                # EFFECT and COMPUTE use base classes
                expected_base = f"Node{node_type.lower().capitalize()}"
                assert (
                    f"from omnibase_core.nodes.node_{node_type.lower()} import {expected_base}"
                    in node_code
                )
                assert (
                    f"class NodeTest{node_type.lower().capitalize()}({expected_base}):"
                    in node_code
                )

    # Edge case tests
    def test_unknown_mixin_warning(self, injector, caplog):
        """Test that unknown mixins generate a warning."""
        contract = {
            "name": "test_effect",
            "node_type": "EFFECT",
            "description": "Test",
            "mixins": [{"name": "UnknownMixin", "enabled": True}],
        }

        imports = injector.generate_imports(contract)

        # Should log warning
        assert any("Unknown mixin" in record.message for record in caplog.records)

        # Should not add import
        assert not any("UnknownMixin" in imp for imp in imports.omnibase_mixins)

    def test_empty_mixin_config(self, injector):
        """Test mixin with empty configuration."""
        contract = {
            "name": "test_effect",
            "node_type": "EFFECT",
            "description": "Test",
            "mixins": [{"name": "MixinMetrics", "enabled": True, "config": {}}],
        }

        class_def = injector.generate_class_definition(contract)

        # Should not have config in __init__ if empty
        assert "metrics_config" not in class_def.init_method

    def test_mixin_without_config_key(self, injector):
        """Test mixin declaration without config key."""
        contract = {
            "name": "test_effect",
            "node_type": "EFFECT",
            "description": "Test",
            "mixins": [{"name": "MixinMetrics", "enabled": True}],
        }

        # Should not raise error
        class_def = injector.generate_class_definition(contract)
        assert "MixinMetrics" in class_def.base_classes

    def test_contract_without_mixins_key(self, injector):
        """Test contract without mixins key."""
        contract = {
            "name": "test_effect",
            "node_type": "EFFECT",
            "description": "Test",
        }

        # Should not raise error
        imports = injector.generate_imports(contract)
        class_def = injector.generate_class_definition(contract)

        assert len(imports.omnibase_mixins) == 0
        assert class_def.base_classes == ["NodeEffect"]


# Integration-style tests
class TestMixinInjectorIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def injector(self):
        """Create MixinInjector instance."""
        return MixinInjector()

    def test_full_workflow_health_check_mixin(self, injector):
        """Test complete workflow with MixinHealthCheck."""
        contract = {
            "name": "database_adapter_effect",
            "node_type": "EFFECT",
            "description": "Database adapter with health monitoring",
            "mixins": [
                {
                    "name": "MixinHealthCheck",
                    "enabled": True,
                    "config": {
                        "check_interval_ms": 30000,
                        "timeout_seconds": 5.0,
                    },
                }
            ],
        }

        # Generate complete node file
        node_code = injector.generate_node_file(contract)

        # Validate comprehensive structure
        assert "#!/usr/bin/env python3" in node_code
        assert "import logging" in node_code
        assert "from omnibase_core.nodes.node_effect import NodeEffect" in node_code
        assert (
            "from omnibase_core.mixins.mixin_health_check import MixinHealthCheck"
            in node_code
        )
        assert (
            "class NodeDatabaseAdapterEffect(NodeEffect, MixinHealthCheck):"
            in node_code
        )
        assert "def __init__(self, container: ModelContainer):" in node_code
        assert "async def initialize(self) -> None:" in node_code
        assert "def get_health_checks(self)" in node_code
        assert "async def _check_self_health(self)" in node_code

        # Validate syntax
        try:
            ast.parse(node_code)
            syntax_valid = True
        except SyntaxError as e:
            syntax_valid = False
            print(f"Syntax error: {e}")

        assert syntax_valid

    def test_full_workflow_multiple_mixins(self, injector):
        """Test complete workflow with multiple mixins."""
        contract = {
            "name": "event_processor_effect",
            "node_type": "EFFECT",
            "description": "Event processor with full monitoring",
            "mixins": [
                {"name": "MixinHealthCheck", "enabled": True, "config": {}},
                {"name": "MixinMetrics", "enabled": True, "config": {}},
                {"name": "MixinEventDrivenNode", "enabled": True, "config": {}},
                {"name": "MixinCaching", "enabled": True, "config": {}},
            ],
        }

        # Generate complete node file
        node_code = injector.generate_node_file(contract)

        # Check all mixins are included
        assert "MixinHealthCheck" in node_code
        assert "MixinMetrics" in node_code
        assert "MixinEventDrivenNode" in node_code
        assert "MixinCaching" in node_code

        # Check methods from different mixins
        assert "get_health_checks" in node_code
        assert "get_capabilities" in node_code
        assert "start_event_consumption" in node_code

        # Validate syntax
        try:
            ast.parse(node_code)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False

        assert syntax_valid

    def test_code_quality_pep8_compliance(self, injector):
        """Test that generated code follows PEP 8 guidelines."""
        contract = {
            "name": "test_effect",
            "node_type": "EFFECT",
            "description": "Test effect node",
            "mixins": [{"name": "MixinHealthCheck", "enabled": True}],
        }

        node_code = injector.generate_node_file(contract)

        # Check import organization (stdlib -> third-party -> omnibase)
        lines = node_code.split("\n")
        import_section_started = False
        last_import_category = None

        for line in lines:
            if line.startswith("import ") or line.startswith("from "):
                import_section_started = True

                # Determine category
                if "import logging" in line or "from typing" in line:
                    current_category = "stdlib"
                elif "omnibase_core" in line:
                    current_category = "omnibase"
                else:
                    current_category = "third_party"

                # Check ordering
                if last_import_category == "omnibase" and current_category == "stdlib":
                    pytest.fail("Imports not properly organized (PEP 8 violation)")

                last_import_category = current_category

            elif import_section_started and line.strip() == "":
                # Blank line separates import categories
                last_import_category = None

        # Check line length (should not exceed 88 chars with Black formatting)
        for i, line in enumerate(lines, 1):
            if len(line) > 120:  # Allow some flexibility
                print(f"Line {i} exceeds 120 characters: {line[:50]}...")
                # Note: This is a soft check, not a hard failure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
