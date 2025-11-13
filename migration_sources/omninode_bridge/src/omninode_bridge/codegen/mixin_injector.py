#!/usr/bin/env python3
"""
Mixin Injector for ONEX Code Generation.

Generates Python code that incorporates omnibase_core mixins into node implementations.
Takes parsed contract data and produces complete node.py files with:
- Import statements for mixins
- Class inheritance chain (NodeEffect + Mixins)
- Mixin initialization code
- Mixin configuration setup
- Required mixin methods

ONEX v2.0 Compliance:
- Generates PEP 8 compliant code
- Includes comprehensive type hints
- Follows omnibase_core patterns
- Produces production-ready implementations

Example:
    >>> from omninode_bridge.codegen.mixin_injector import MixinInjector
    >>> injector = MixinInjector()
    >>> imports = injector.generate_imports(contract)
    >>> class_def = injector.generate_class_definition(contract)
    >>> node_code = injector.generate_node_file(contract)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelGeneratedImports:
    """
    Generated import statements organized by category.

    Imports are organized into categories for proper PEP 8 formatting:
    - Standard library imports (e.g., os, sys, typing)
    - Third-party imports (e.g., httpx, asyncpg)
    - omnibase_core imports (e.g., NodeEffect, models)
    - omnibase mixin imports (e.g., MixinHealthCheck)
    - Project-local imports (e.g., from .models import X)

    Attributes:
        standard_library: Standard library imports
        third_party: Third-party package imports
        omnibase_core: omnibase_core base imports
        omnibase_mixins: omnibase_core mixin imports
        project_local: Local project imports
    """

    standard_library: list[str] = field(default_factory=list)
    third_party: list[str] = field(default_factory=list)
    omnibase_core: list[str] = field(default_factory=list)
    omnibase_mixins: list[str] = field(default_factory=list)
    project_local: list[str] = field(default_factory=list)


@dataclass
class ModelGeneratedClass:
    """
    Generated class definition components.

    Contains all components needed to construct a complete Python class:
    - Class name and inheritance
    - Docstring
    - __init__ method
    - async initialize() method
    - Additional methods

    Attributes:
        class_name: Name of the generated class (e.g., NodeMyEffect)
        base_classes: List of base class names (NodeEffect + mixins)
        docstring: Class-level docstring
        init_method: __init__ method implementation
        initialize_method: async initialize() method implementation
        shutdown_method: async shutdown() method implementation
        methods: Additional method implementations
    """

    class_name: str
    base_classes: list[str]
    docstring: str
    init_method: str
    initialize_method: str
    shutdown_method: str = ""
    methods: list[str] = field(default_factory=list)


# Mixin catalog for reference
MIXIN_CATALOG = {
    # Health & Monitoring
    "MixinHealthCheck": {
        "import_path": "omnibase_core.mixins.mixin_health_check",
        "dependencies": [],
        "required_methods": ["get_health_checks"],
        "description": "Health check implementation with async support",
        "default_config": {
            "check_interval_ms": 60000,
            "timeout_seconds": 10.0,
            "components": [
                {
                    "name": "service",
                    "critical": True,
                    "timeout_seconds": 5.0,
                }
            ],
        },
    },
    "MixinMetrics": {
        "import_path": "omnibase_core.mixins.mixin_metrics",
        "dependencies": [],
        "required_methods": [],
        "description": "Performance metrics collection",
        "default_config": {
            "metrics_prefix": "node",
            "collect_latency": True,
            "collect_throughput": True,
            "collect_error_rates": True,
            "percentiles": [50, 95, 99],
            "histogram_buckets": [100, 500, 1000, 2000, 5000],
        },
    },
    "MixinLogData": {
        "import_path": "omnibase_core.mixins.mixin_log_data",
        "dependencies": [],
        "required_methods": [],
        "description": "Structured logging data model",
        "default_config": {
            "structured": True,
            "json_format": True,
            "correlation_tracking": True,
        },
    },
    "MixinRequestResponseIntrospection": {
        "import_path": "omnibase_core.mixins.mixin_request_response_introspection",
        "dependencies": [],
        "required_methods": [],
        "description": "Request/response tracking for discovery",
        "default_config": {
            "track_requests": True,
            "track_responses": True,
            "sample_rate": 1.0,
        },
    },
    # Event-Driven Patterns
    "MixinEventDrivenNode": {
        "import_path": "omnibase_core.mixins.mixin_event_driven_node",
        "dependencies": [
            "MixinEventHandler",
            "MixinNodeLifecycle",
            "MixinIntrospectionPublisher",
        ],
        "required_methods": ["get_capabilities", "supports_introspection"],
        "description": "Complete event-driven node capabilities",
        "default_config": {
            "event_driven": True,
            "supports_introspection": True,
        },
    },
    "MixinEventBus": {
        "import_path": "omnibase_core.mixins.mixin_event_bus",
        "dependencies": [],
        "required_methods": ["get_event_patterns"],
        "description": "Event bus operations and publishing",
        "default_config": {
            "event_patterns": ["*"],
            "async_publishing": True,
        },
    },
    "MixinEventHandler": {
        "import_path": "omnibase_core.mixins.mixin_event_handler",
        "dependencies": [],
        "required_methods": [],
        "description": "Event handler registration and routing",
        "default_config": {
            "routing_enabled": True,
            "async_handlers": True,
        },
    },
    # Service Integration
    "MixinServiceRegistry": {
        "import_path": "omnibase_core.mixins.mixin_service_registry",
        "dependencies": [],
        "required_methods": [],
        "description": "Service discovery and registration",
        "default_config": {
            "auto_register": True,
            "heartbeat_interval_ms": 30000,
        },
    },
    # Data Handling
    "MixinCaching": {
        "import_path": "omnibase_core.mixins.mixin_caching",
        "dependencies": [],
        "required_methods": [],
        "description": "Result caching for expensive operations",
        "default_config": {
            "cache_ttl_seconds": 300,
            "max_cache_size": 1000,
            "eviction_policy": "lru",
        },
    },
    "MixinHashComputation": {
        "import_path": "omnibase_core.mixins.mixin_hash_computation",
        "dependencies": [],
        "required_methods": [],
        "description": "Hash computation utilities",
        "default_config": {
            "algorithm": "blake3",
            "buffer_size": 65536,
        },
    },
    # Serialization
    "MixinCanonicalYAMLSerializer": {
        "import_path": "omnibase_core.mixins.mixin_canonical_serialization",
        "dependencies": [],
        "required_methods": [],
        "description": "Canonical YAML serialization",
        "default_config": {
            "sort_keys": True,
            "canonical": True,
            "indent": 2,
        },
    },
}

# Convenience wrapper catalog
# Maps node types to their convenience wrapper classes and standard mixins
CONVENIENCE_WRAPPER_CATALOG = {
    "orchestrator": {
        "class_name": "ModelServiceOrchestrator",
        "import_path": "omninode_bridge.utils.node_services",
        "standard_mixins": [
            "MixinNodeService",
            "MixinHealthCheck",
            "MixinEventBus",
            "MixinMetrics",
        ],
        "description": "Pre-composed orchestrator with standard mixins",
    },
    "reducer": {
        "class_name": "ModelServiceReducer",
        "import_path": "omninode_bridge.utils.node_services",
        "standard_mixins": [
            "MixinNodeService",
            "MixinHealthCheck",
            "MixinCaching",
            "MixinMetrics",
        ],
        "description": "Pre-composed reducer with standard mixins",
    },
}


class MixinInjector:
    """
    Generate Python code with mixin inheritance.

    Responsibilities:
    - Generate import statements for mixins
    - Generate class inheritance chain
    - Generate mixin initialization code
    - Generate configuration setup
    - Generate required mixin methods

    Thread-safe and stateless - can be instantiated once and reused.

    Example:
        >>> injector = MixinInjector()
        >>> imports = injector.generate_imports(contract)
        >>> class_def = injector.generate_class_definition(contract)
        >>> node_code = injector.generate_node_file(contract)
    """

    def __init__(self):
        """Initialize MixinInjector."""
        self.mixin_catalog = MIXIN_CATALOG
        self.convenience_wrapper_catalog = CONVENIENCE_WRAPPER_CATALOG

    def _ensure_dict(self, contract: dict[str, Any] | Any) -> dict[str, Any]:
        """
        Ensure contract is a dict, converting from Pydantic model or dataclass if needed.

        Args:
            contract: Contract data as dict, Pydantic model, or dataclass

        Returns:
            Contract as dict
        """
        from dataclasses import asdict, is_dataclass

        # If it's already a dict, return it
        if isinstance(contract, dict):
            return contract

        # If it's a dataclass, convert to dict
        if is_dataclass(contract):
            return asdict(contract)

        # If it's a Pydantic model, convert to dict
        if hasattr(contract, "model_dump"):
            return contract.model_dump()
        elif hasattr(contract, "dict"):
            return contract.dict()

        # Unknown type - return as-is and hope for the best
        return contract

    def _should_use_convenience_wrapper(self, contract: dict[str, Any]) -> bool:
        """
        Determine if contract should use a convenience wrapper.

        A convenience wrapper should be used if:
        1. Node type has a convenience wrapper available
        2. Contract uses standard mixins (or no mixins specified)
        3. No custom mixin configurations are specified

        Args:
            contract: Contract data with node type and mixin declarations

        Returns:
            True if convenience wrapper should be used, False for custom composition
        """
        node_type = contract.get("node_type", "").lower()

        # Check if convenience wrapper exists for this node type
        if node_type not in self.convenience_wrapper_catalog:
            return False

        wrapper_info = self.convenience_wrapper_catalog[node_type]
        standard_mixins = set(wrapper_info["standard_mixins"])

        # Get declared mixins from contract
        declared_mixins = contract.get("mixins", [])
        enabled_mixin_names = {
            m.get("name", "") for m in declared_mixins if m.get("enabled", True)
        }

        # If no mixins declared, use convenience wrapper with defaults
        if not enabled_mixin_names:
            return True

        # Check if declared mixins match standard mixins exactly
        # If they have the same mixins, use convenience wrapper
        if enabled_mixin_names == standard_mixins:
            # Also check that no custom configurations are specified
            has_custom_config = any(
                m.get("config") for m in declared_mixins if m.get("enabled", True)
            )
            if not has_custom_config:
                return True

        return False

    def _get_convenience_wrapper_info(self, node_type: str) -> dict[str, Any] | None:
        """
        Get convenience wrapper information for node type.

        Args:
            node_type: Node type (effect, compute, reducer, orchestrator)

        Returns:
            Convenience wrapper info dict or None if not available
        """
        return self.convenience_wrapper_catalog.get(node_type.lower())

    def generate_imports(self, contract: dict[str, Any]) -> ModelGeneratedImports:
        """
        Generate import statements for node file.

        Args:
            contract: Contract data with mixin declarations

        Returns:
            Organized import statements

        Example:
            >>> imports = injector.generate_imports({
            ...     "mixins": [
            ...         {"name": "MixinHealthCheck", "enabled": True},
            ...         {"name": "MixinMetrics", "enabled": True}
            ...     ]
            ... })
            >>> print(imports.omnibase_mixins)
            ['from omnibase_core.mixins.mixin_health_check import MixinHealthCheck',
             'from omnibase_core.mixins.mixin_metrics import MixinMetrics']
        """
        # Ensure contract is a dict (convert from Pydantic if needed)
        contract = self._ensure_dict(contract)

        imports = ModelGeneratedImports()

        # Standard library imports (always needed)
        imports.standard_library.extend(
            [
                "import logging",
                "from typing import Any, Optional",
            ]
        )

        # Always import NodeEffect base class
        node_type = contract.get("node_type", "EFFECT").lower()
        if node_type == "effect":
            imports.omnibase_core.append(
                "from omnibase_core.nodes.node_effect import NodeEffect"
            )
        elif node_type == "compute":
            imports.omnibase_core.append(
                "from omnibase_core.nodes.node_compute import NodeCompute"
            )
        elif node_type == "reducer":
            imports.omnibase_core.append(
                "from omnibase_core.nodes.node_reducer import NodeReducer"
            )
        elif node_type == "orchestrator":
            imports.omnibase_core.append(
                "from omnibase_core.nodes.node_orchestrator import NodeOrchestrator"
            )

        # Import ModelContainer
        imports.omnibase_core.append(
            "from omnibase_core.models.core.model_container import ModelContainer"
        )

        # Import contract model
        contract_class_name = f"ModelContract{node_type.capitalize()}"
        imports.omnibase_core.append(
            f"from omnibase_core.models.contracts.model_contract_{node_type} import {contract_class_name}"
        )

        # Get mixins list early so it's available for all code paths
        mixins = contract.get("mixins", [])

        # Check if we should use convenience wrapper
        use_convenience_wrapper = self._should_use_convenience_wrapper(contract)

        if use_convenience_wrapper:
            # Import convenience wrapper instead of individual mixins
            wrapper_info = self._get_convenience_wrapper_info(node_type)
            if wrapper_info:
                wrapper_class = wrapper_info["class_name"]
                import_path = wrapper_info["import_path"]
                imports.project_local.append(
                    f"from {import_path} import {wrapper_class}"
                )
                logger.debug(
                    f"Using convenience wrapper: {wrapper_class} for {node_type}"
                )
        else:
            # Import individual mixins (existing behavior)
            for mixin_decl in mixins:
                if not mixin_decl.get("enabled", True):
                    continue

                mixin_name = mixin_decl.get("name", "")
                if mixin_name in self.mixin_catalog:
                    import_path = self.mixin_catalog[mixin_name]["import_path"]
                    imports.omnibase_mixins.append(
                        f"from {import_path} import {mixin_name}"
                    )
                else:
                    logger.warning(f"Unknown mixin: {mixin_name}")

        # Import circuit breaker if enabled
        advanced_features = contract.get("advanced_features") or {}
        if advanced_features.get("circuit_breaker", {}).get("enabled"):
            imports.omnibase_core.append(
                "from omnibase_core.nodes.model_circuit_breaker import ModelCircuitBreaker"
            )

        # Import health status models if MixinHealthCheck enabled
        if any(
            m.get("name") == "MixinHealthCheck" and m.get("enabled", True)
            for m in mixins
        ):
            imports.omnibase_core.extend(
                [
                    "from omnibase_core.models.core.model_health_status import ModelHealthStatus",
                    "from omnibase_core.enums.enum_node_health_status import EnumNodeHealthStatus",
                ]
            )

        logger.debug(f"Generated {len(imports.omnibase_mixins)} mixin imports")
        return imports

    def generate_class_definition(
        self, contract: dict[str, Any]
    ) -> ModelGeneratedClass:
        """
        Generate class definition with mixin inheritance.

        Args:
            contract: Contract data with mixin declarations

        Returns:
            Generated class code components

        Example:
            >>> class_def = injector.generate_class_definition({
            ...     "name": "postgres_crud_effect",
            ...     "node_type": "EFFECT",
            ...     "description": "PostgreSQL CRUD operations",
            ...     "mixins": [{"name": "MixinHealthCheck", "enabled": True}]
            ... })
            >>> print(class_def.class_name)
            'NodePostgresCrudEffect'
            >>> print(class_def.base_classes)
            ['NodeEffect', 'MixinHealthCheck']
        """
        # Ensure contract is a dict (convert from Pydantic if needed)
        contract = self._ensure_dict(contract)

        # Determine node name from contract
        service_name = contract.get("name", "UnknownNode")
        # Convert snake_case to PascalCase
        class_name_parts = [part.capitalize() for part in service_name.split("_")]
        class_name = "Node" + "".join(class_name_parts)

        # Build inheritance chain: Convenience wrapper OR NodeBase + Mixins
        node_type = contract.get("node_type", "EFFECT").lower()
        use_convenience_wrapper = self._should_use_convenience_wrapper(contract)

        if use_convenience_wrapper:
            # Use convenience wrapper as sole base class
            wrapper_info = self._get_convenience_wrapper_info(node_type)
            if wrapper_info:
                base_classes = [wrapper_info["class_name"]]
                logger.debug(
                    f"Using convenience wrapper inheritance: {wrapper_info['class_name']}"
                )
            else:
                # Fallback to traditional pattern if wrapper info not found
                base_node = f"Node{node_type.capitalize()}"
                base_classes = [base_node]
        else:
            # Traditional pattern: NodeBase + Mixins
            base_node = f"Node{node_type.capitalize()}"
            base_classes = [base_node]

            mixins = contract.get("mixins", [])
            for mixin_decl in mixins:
                if mixin_decl.get("enabled", True):
                    mixin_name = mixin_decl.get("name", "")
                    if mixin_name:
                        base_classes.append(mixin_name)

        # Generate components based on whether we're using convenience wrapper
        docstring = self._generate_docstring(contract)

        if use_convenience_wrapper:
            # Wrapper provides all mixin implementations - generate minimal class
            init_method = (
                "    def __init__(self, container: ModelContainer):\n"
                '        """Initialize node with container."""\n'
                "        super().__init__(container)\n"
                "        \n"
                "        # Initialize logger\n"
                "        self.logger = logging.getLogger(self.__class__.__name__)"
            )
            initialize_method = (
                "    async def initialize(self) -> None:\n"
                '        """Initialize node resources and mixins."""\n'
                "        # Initialize base class and wrapper mixins\n"
                "        await super().initialize()\n"
                "        \n"
                "        self.logger.info(f'Initializing {self.__class__.__name__}')\n"
                "        \n"
                "        self.logger.info(f'{self.__class__.__name__} initialized successfully')"
            )
            shutdown_method = (
                "    async def shutdown(self) -> None:\n"
                '        """Shutdown node and cleanup resources."""\n'
                "        self.logger.info(f'Shutting down {self.__class__.__name__}')\n"
                "        \n"
                f"        # Shutdown base Node{node_type.capitalize()}\n"
                "        await super().shutdown()"
            )
            methods = []  # No stub methods - wrapper provides implementations
        else:
            # Traditional pattern - generate full implementation with stubs
            init_method = self._generate_init_method(contract)
            initialize_method = self._generate_initialize_method(contract)
            shutdown_method = self._generate_shutdown_method(contract)
            methods = self._generate_mixin_methods(contract)

        return ModelGeneratedClass(
            class_name=class_name,
            base_classes=base_classes,
            docstring=docstring,
            init_method=init_method,
            initialize_method=initialize_method,
            shutdown_method=shutdown_method,
            methods=methods,
        )

    def _generate_docstring(self, contract: dict[str, Any]) -> str:
        """
        Generate comprehensive class docstring.

        Args:
            contract: Contract data

        Returns:
            Formatted docstring
        """
        description = contract.get("description", "ONEX v2.0 Node")
        node_type = contract.get("node_type", "EFFECT")
        use_convenience_wrapper = self._should_use_convenience_wrapper(contract)

        lines = [
            '    """',
            f"    {description}",
            "",
            f"    ONEX v2.0 Compliant {node_type.capitalize()} Node",
            "",
        ]

        if use_convenience_wrapper:
            wrapper_info = self._get_convenience_wrapper_info(node_type)
            if wrapper_info:
                lines.append(
                    f"    Base Class: {wrapper_info['class_name']} ({wrapper_info['description']})"
                )
                lines.append("")
                lines.append("    Pre-configured Capabilities:")
                for mixin in wrapper_info["standard_mixins"]:
                    if mixin in self.mixin_catalog:
                        desc = self.mixin_catalog[mixin]["description"]
                        lines.append(f"        - {mixin}: {desc}")
        else:
            lines.append("    Capabilities:")
            # NodeEffect built-in features
            base_type = f"Node{node_type.lower().capitalize()}"
            lines.append(f"      Built-in Features ({base_type}):")
            lines.append("        - Circuit breakers with failure threshold")
            lines.append("        - Retry policies with exponential backoff")
            lines.append("        - Transaction support with rollback")
            lines.append("        - Concurrent execution control")
            lines.append("        - Performance metrics tracking")

            # Mixin features
            mixins = contract.get("mixins", [])
            enabled_mixins = [m for m in mixins if m.get("enabled", True)]

            if enabled_mixins:
                lines.append("")
                lines.append("      Enhanced Features (Mixins):")
                for mixin in enabled_mixins:
                    mixin_name = mixin.get("name", "")
                    if mixin_name in self.mixin_catalog:
                        desc = self.mixin_catalog[mixin_name]["description"]
                        lines.append(f"        - {mixin_name}: {desc}")

        lines.append('    """')
        return "\n".join(lines)

    def _generate_init_method(self, contract: dict[str, Any]) -> str:
        """
        Generate __init__ method with mixin initialization.

        Args:
            contract: Contract data

        Returns:
            __init__ method code
        """
        lines = [
            "    def __init__(self, container: ModelContainer):",
            '        """Initialize node with container and mixins."""',
            "        # Initialize base classes (Node + Mixins)",
            "        super().__init__(container)",
            "        ",
            "        # Initialize logger",
            "        self.logger = logging.getLogger(self.__class__.__name__)",
        ]

        # Generate mixin configuration
        mixins = contract.get("mixins", [])
        for mixin in mixins:
            if not mixin.get("enabled", True):
                continue

            mixin_name = mixin.get("name", "")
            mixin_config = mixin.get("config", {})

            if mixin_config:
                config_var = f"{mixin_name.lower().replace('mixin', '')}_config"
                lines.append("")
                lines.append(f"        # Configure {mixin_name}")
                lines.append(f"        self.{config_var} = {{")
                for key, value in mixin_config.items():
                    lines.append(f'            "{key}": {value!r},')
                lines.append("        }")

        return "\n".join(lines)

    def _generate_initialize_method(self, contract: dict[str, Any]) -> str:
        """
        Generate async initialize() method.

        Calls super().initialize() and mixin setup methods.

        Args:
            contract: Contract data

        Returns:
            initialize() method code
        """
        node_type = contract.get("node_type", "EFFECT").lower()
        base_class = f"Node{node_type.capitalize()}"

        lines = [
            "    async def initialize(self) -> None:",
            '        """Initialize node resources and mixins."""',
            f"        # Initialize base {base_class}",
            "        await super().initialize()",
            "        ",
            "        self.logger.info(f'Initializing {self.__class__.__name__}')",
        ]

        # Initialize mixins
        mixins = contract.get("mixins", [])
        for mixin in mixins:
            if not mixin.get("enabled", True):
                continue

            mixin_name = mixin.get("name", "")

            # MixinHealthCheck setup
            if mixin_name == "MixinHealthCheck":
                lines.append("")
                lines.append("        # Setup health checks")
                lines.append("        health_checks = self.get_health_checks()")
                lines.append("        for check_name, check_func in health_checks:")
                lines.append(
                    "            self.register_health_check(check_name, check_func)"
                )

            # MixinEventDrivenNode setup
            elif mixin_name == "MixinEventDrivenNode":
                lines.append("")
                lines.append("        # Setup event consumption")
                lines.append("        await self.start_event_consumption()")

            # MixinServiceRegistry setup
            elif mixin_name == "MixinServiceRegistry":
                lines.append("")
                lines.append("        # Start service registry")
                domain_filter = mixin.get("config", {}).get("domain_filter")
                if domain_filter:
                    lines.append(
                        f'        self.start_service_registry(domain_filter="{domain_filter}")'
                    )
                else:
                    lines.append("        self.start_service_registry()")

        lines.append("")
        lines.append(
            "        self.logger.info(f'{self.__class__.__name__} initialized successfully')"
        )

        return "\n".join(lines)

    def _generate_shutdown_method(self, contract: dict[str, Any]) -> str:
        """
        Generate async shutdown() method.

        Calls mixin cleanup and super().shutdown().

        Args:
            contract: Contract data

        Returns:
            shutdown() method code
        """
        node_type = contract.get("node_type", "EFFECT").lower()

        lines = [
            "    async def shutdown(self) -> None:",
            '        """Shutdown node and cleanup resources."""',
            "        self.logger.info(f'Shutting down {self.__class__.__name__}')",
            "",
        ]

        # Cleanup mixins
        mixins = contract.get("mixins", [])
        has_cleanup = False

        for mixin in mixins:
            if not mixin.get("enabled", True):
                continue

            mixin_name = mixin.get("name", "")

            # MixinEventDrivenNode cleanup
            if mixin_name == "MixinEventDrivenNode":
                if not has_cleanup:
                    lines.append("        # Cleanup mixins")
                    has_cleanup = True
                lines.append("        await self.stop_event_consumption()")

            # MixinNodeLifecycle cleanup
            elif mixin_name == "MixinNodeLifecycle":
                if not has_cleanup:
                    lines.append("        # Cleanup mixins")
                    has_cleanup = True
                lines.append("        self.cleanup_lifecycle_resources()")

        if has_cleanup:
            lines.append("")

        lines.append(f"        # Shutdown base Node{node_type.capitalize()}")
        lines.append("        await super().shutdown()")

        return "\n".join(lines)

    def _generate_mixin_methods(self, contract: dict[str, Any]) -> list[str]:
        """
        Generate mixin-required methods.

        Args:
            contract: Contract data

        Returns:
            List of method implementations
        """
        methods = []
        mixins = contract.get("mixins", [])

        for mixin in mixins:
            if not mixin.get("enabled", True):
                continue

            mixin_name = mixin.get("name", "")

            # MixinHealthCheck requires get_health_checks and check methods
            if mixin_name == "MixinHealthCheck":
                method = '''    def get_health_checks(self) -> list[tuple[str, Any]]:
        """
        Register health checks for this node.

        Returns:
            List of (check_name, check_function) tuples
        """
        return [
            ("self", self._check_self_health),
            # TODO: Add additional health checks as needed
        ]

    async def _check_self_health(self) -> ModelHealthStatus:
        """
        Check node's own health status.

        Returns:
            Health status model
        """
        try:
            # TODO: Implement actual health check logic
            # Example: Check database connection, API availability, etc.

            return ModelHealthStatus(
                status=EnumNodeHealthStatus.HEALTHY,
                message="Node is healthy",
            )
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return ModelHealthStatus(
                status=EnumNodeHealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
            )'''
                methods.append(method)

            # MixinEventDrivenNode requires capabilities
            elif mixin_name == "MixinEventDrivenNode":
                method = '''    def get_capabilities(self) -> list[str]:
        """
        Define node capabilities for service discovery.

        Returns:
            List of capability names
        """
        # TODO: Define actual capabilities
        return [
            "data_processing",
            "transformation",
            # Add more capabilities as needed
        ]

    def supports_introspection(self) -> bool:
        """
        Check if this node supports introspection.

        Returns:
            True if introspection is supported
        """
        return True'''
                methods.append(method)

            # MixinEventBus requires event patterns
            elif mixin_name == "MixinEventBus":
                method = '''    def get_event_patterns(self) -> list[str]:
        """
        Define event patterns this node listens for.

        Returns:
            List of event pattern strings
        """
        # TODO: Define actual event patterns
        return [
            "*.*.start",
            "*.*.process",
            # Add more patterns as needed
        ]'''
                methods.append(method)

        return methods

    def generate_node_file(self, contract: dict[str, Any]) -> str:
        """
        Generate complete node.py file with mixins.

        Args:
            contract: Contract data with mixin declarations

        Returns:
            Complete node.py file content

        Example:
            >>> node_code = injector.generate_node_file({
            ...     "name": "postgres_crud_effect",
            ...     "node_type": "EFFECT",
            ...     "description": "PostgreSQL CRUD operations",
            ...     "mixins": [
            ...         {"name": "MixinHealthCheck", "enabled": True},
            ...         {"name": "MixinMetrics", "enabled": True}
            ...     ]
            ... })
            >>> assert "class NodePostgresCrudEffect" in node_code
            >>> assert "MixinHealthCheck" in node_code
        """
        # Ensure contract is a dict (convert from Pydantic if needed)
        contract = self._ensure_dict(contract)

        imports = self.generate_imports(contract)
        class_def = self.generate_class_definition(contract)

        # Build file content
        lines = [
            "#!/usr/bin/env python3",
            '"""',
            f'{contract.get("description", "ONEX v2.0 Node")}',
            "",
            "Generated by OmniNode Code Generator",
            "DO NOT EDIT MANUALLY - Regenerate from contract",
            '"""',
            "",
        ]

        # Add imports (sorted and organized)
        if imports.standard_library:
            lines.extend(sorted(imports.standard_library))
            lines.append("")

        if imports.third_party:
            lines.extend(sorted(imports.third_party))
            lines.append("")

        if imports.omnibase_core:
            lines.extend(sorted(imports.omnibase_core))
            lines.append("")

        if imports.omnibase_mixins:
            lines.extend(sorted(imports.omnibase_mixins))
            lines.append("")

        if imports.project_local:
            lines.extend(sorted(imports.project_local))
            lines.append("")

        # Setup logger
        lines.append("logger = logging.getLogger(__name__)")
        lines.append("")
        lines.append("")

        # Add class definition
        inheritance = ", ".join(class_def.base_classes)
        lines.append(f"class {class_def.class_name}({inheritance}):")
        lines.append(class_def.docstring)
        lines.append("")
        lines.append(class_def.init_method)
        lines.append("")
        lines.append(class_def.initialize_method)

        # Add shutdown method
        if class_def.shutdown_method:
            lines.append("")
            lines.append(class_def.shutdown_method)

        # Add methods
        for method in class_def.methods:
            lines.append("")
            lines.append(method)

        return "\n".join(lines)


__all__ = [
    "ModelGeneratedImports",
    "ModelGeneratedClass",
    "MixinInjector",
]
