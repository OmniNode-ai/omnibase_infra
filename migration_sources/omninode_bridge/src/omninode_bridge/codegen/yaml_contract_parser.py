#!/usr/bin/env python3
"""
YAML Contract Parser for ONEX v2.0 Code Generation.

Parses YAML contract files and validates mixin declarations.

Features:
- Parse v1.0 and v2.0 contracts
- JSON Schema validation
- Mixin registry loading from catalog
- Mixin dependency checking
- Configuration validation
- Backward compatibility with v1.0 contracts

Thread-safe and stateless.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, ClassVar, Optional

import yaml
from jsonschema import RefResolver, ValidationError, validate

try:
    # Try relative import first (when used as package)
    from .models_contract import (
        EnumFallbackStrategy,
        EnumLLMTier,
        EnumQualityLevel,
        EnumTemplateVariant,
        ModelAdvancedFeatures,
        ModelCircuitBreakerConfig,
        ModelDeadLetterQueueConfig,
        ModelEnhancedContract,
        ModelGenerationDirectives,
        ModelMixinDeclaration,
        ModelObservabilityConfig,
        ModelQualityGate,
        ModelQualityGatesConfiguration,
        ModelRetryPolicyConfig,
        ModelSecurityValidationConfig,
        ModelTemplateConfiguration,
        ModelTransactionsConfig,
        ModelVersionInfo,
    )
except ImportError:
    # Fall back to direct import (when used standalone)
    from models_contract import (  # type: ignore
        EnumFallbackStrategy,
        EnumLLMTier,
        EnumQualityLevel,
        EnumTemplateVariant,
        ModelAdvancedFeatures,
        ModelCircuitBreakerConfig,
        ModelDeadLetterQueueConfig,
        ModelEnhancedContract,
        ModelGenerationDirectives,
        ModelMixinDeclaration,
        ModelObservabilityConfig,
        ModelQualityGate,
        ModelQualityGatesConfiguration,
        ModelRetryPolicyConfig,
        ModelSecurityValidationConfig,
        ModelTemplateConfiguration,
        ModelTransactionsConfig,
        ModelVersionInfo,
    )

logger = logging.getLogger(__name__)


class YAMLContractParser:
    """
    Parse and validate YAML contracts with mixin support.

    Provides comprehensive contract parsing with:
    - JSON Schema validation
    - Mixin registry lookup
    - Dependency checking
    - Configuration validation
    - Backward compatibility

    Thread-safe and stateless - can be reused across calls.

    Example:
        >>> parser = YAMLContractParser()
        >>> contract = parser.parse_contract_file("path/to/contract.yaml")
        >>> if contract.is_valid:
        ...     print(f"Contract: {contract.name}")
        ...     print(f"Mixins: {contract.get_mixin_names()}")
        ... else:
        ...     print(f"Errors: {contract.validation_errors}")
    """

    # Mixin registry: name -> metadata
    # Populated from mixin catalog on first use
    _mixin_registry: ClassVar[dict[str, dict[str, Any]]] = {}

    # Mixin dependencies: mixin_name -> list of required mixins
    _mixin_dependencies: ClassVar[dict[str, list[str]]] = {
        "MixinEventDrivenNode": [
            "MixinEventHandler",
            "MixinNodeLifecycle",
            "MixinIntrospectionPublisher",
        ],
        "MixinNodeExecutor": ["MixinEventDrivenNode"],
    }

    # NodeEffect built-in features (don't need mixins for these)
    _nodeeffect_builtin_features: ClassVar[set[str]] = {
        "circuit_breaker",
        "retry_policy",
        "transactions",
        "timeout_management",
        "concurrent_execution",
        "performance_metrics",  # Basic metrics; MixinMetrics provides comprehensive
        "event_emission",
        "file_operations",
    }

    def __init__(self, schema_dir: Optional[Path] = None):
        """
        Initialize parser.

        Args:
            schema_dir: Directory containing JSON schemas (defaults to schemas/ in package)
        """
        if schema_dir is None:
            # Default to schemas directory in package
            schema_dir = Path(__file__).parent / "schemas"

        self.schema_dir = schema_dir
        self._load_schemas()
        self._load_mixin_registry()

    def _load_schemas(self) -> None:
        """Load JSON schemas for validation."""
        try:
            # Load main contract schema
            contract_schema_path = self.schema_dir / "contract_schema_v2.json"
            with open(contract_schema_path) as f:
                self.contract_schema = json.load(f)

            # Load mixins schema
            mixins_schema_path = self.schema_dir / "mixins_schema.json"
            with open(mixins_schema_path) as f:
                self.mixins_schema = json.load(f)

            # Load advanced features schema
            advanced_features_schema_path = (
                self.schema_dir / "advanced_features_schema.json"
            )
            with open(advanced_features_schema_path) as f:
                self.advanced_features_schema = json.load(f)

            # Create schema store for $ref resolution
            # Include both relative and absolute URLs
            self.schema_store = {
                # Relative paths
                "mixins_schema.json": self.mixins_schema,
                "advanced_features_schema.json": self.advanced_features_schema,
                # Absolute URLs from schema $id fields
                "https://omninode.dev/schemas/contract/mixins/v1.0.0": self.mixins_schema,
                "https://omninode.dev/schemas/contract/advanced-features/v1.0.0": self.advanced_features_schema,
                # Resolved URLs (base URL + relative path)
                "https://omninode.dev/schemas/contract/mixins_schema.json": self.mixins_schema,
                "https://omninode.dev/schemas/contract/advanced_features_schema.json": self.advanced_features_schema,
                # Main contract schema
                self.contract_schema.get("$id", ""): self.contract_schema,
            }

            # Create RefResolver for schema validation
            self.schema_resolver = RefResolver.from_schema(
                self.contract_schema, store=self.schema_store
            )

            logger.debug("Loaded JSON schemas for validation")

        except FileNotFoundError as e:
            logger.error(f"Failed to load JSON schema: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON schema: {e}")
            raise

    def _load_mixin_registry(self) -> None:
        """
        Load mixin registry from catalog.

        Parses the OMNIBASE_CORE_MIXIN_CATALOG.md to extract:
        - Mixin names
        - Import paths
        - Dependencies
        - NodeEffect overlaps

        This provides the source of truth for available mixins.
        """
        if self._mixin_registry:
            # Already loaded
            return

        # Load catalog markdown file
        catalog_path = (
            Path(__file__).parent.parent.parent.parent
            / "docs"
            / "reference"
            / "OMNIBASE_CORE_MIXIN_CATALOG.md"
        )

        if not catalog_path.exists():
            logger.warning(f"Mixin catalog not found at {catalog_path}")
            # Fall back to hardcoded list
            self._load_fallback_mixin_registry()
            return

        try:
            with open(catalog_path) as f:
                catalog_content = f.read()

            # Parse catalog to extract mixin information
            # Look for patterns like:
            # #### MixinHealthCheck
            # **Module**: `omnibase_core.mixins.mixin_health_check`
            # **Dependencies**: None (or list)
            # **NodeEffect Overlap**: NONE (or PARTIAL)

            mixin_pattern = re.compile(
                r"####\s+(Mixin\w+).*?"
                r"\*\*Module\*\*:\s+`([\w.]+)`.*?"
                r"\*\*Dependencies\*\*:\s+(.+?)(?:\n|$).*?"
                r"\*\*NodeEffect Overlap\*\*:\s+(.+?)(?:\n|$)",
                re.DOTALL,
            )

            matches = mixin_pattern.findall(catalog_content)

            for mixin_name, module_path, dependencies_str, overlap_str in matches:
                # Parse dependencies
                dependencies = []
                if "None" not in dependencies_str and dependencies_str.strip():
                    # Extract mixin names from dependencies string
                    # e.g., "MixinEventHandler, MixinNodeLifecycle, MixinIntrospectionPublisher"
                    dep_matches = re.findall(r"Mixin\w+", dependencies_str)
                    dependencies = dep_matches

                # Parse overlap
                overlap = overlap_str.strip().split()[
                    0
                ]  # Get first word (NONE/PARTIAL)

                self._mixin_registry[mixin_name] = {
                    "module": module_path,
                    "dependencies": dependencies,
                    "nodeeffect_overlap": overlap,
                    "import_path": f"{module_path}.{mixin_name}",
                }

                # Store dependencies for checking
                if dependencies:
                    self._mixin_dependencies[mixin_name] = dependencies

            logger.debug(
                f"Loaded {len(self._mixin_registry)} mixins from catalog: "
                f"{list(self._mixin_registry.keys())}"
            )

        except Exception as e:
            logger.error(f"Failed to parse mixin catalog: {e}")
            self._load_fallback_mixin_registry()

    def _load_fallback_mixin_registry(self) -> None:
        """Load fallback mixin registry with core mixins."""
        # Hardcoded list of commonly used mixins
        core_mixins = [
            ("MixinHealthCheck", "omnibase_core.mixins.mixin_health_check"),
            ("MixinMetrics", "omnibase_core.mixins.mixin_metrics"),
            ("MixinLogData", "omnibase_core.mixins.mixin_log_data"),
            (
                "MixinRequestResponseIntrospection",
                "omnibase_core.mixins.mixin_request_response_introspection",
            ),
            ("MixinEventDrivenNode", "omnibase_core.mixins.mixin_event_driven_node"),
            ("MixinEventBus", "omnibase_core.mixins.mixin_event_bus"),
            ("MixinEventHandler", "omnibase_core.mixins.mixin_event_handler"),
            ("MixinEventListener", "omnibase_core.mixins.mixin_event_listener"),
            (
                "MixinIntrospectionPublisher",
                "omnibase_core.mixins.mixin_introspection_publisher",
            ),
            ("MixinServiceRegistry", "omnibase_core.mixins.mixin_service_registry"),
            (
                "MixinDiscoveryResponder",
                "omnibase_core.mixins.mixin_discovery_responder",
            ),
            ("MixinNodeService", "omnibase_core.mixins.mixin_node_service"),
            ("MixinNodeExecutor", "omnibase_core.mixins.mixin_node_executor"),
            ("MixinNodeLifecycle", "omnibase_core.mixins.mixin_node_lifecycle"),
            ("MixinNodeSetup", "omnibase_core.mixins.mixin_node_setup"),
            ("MixinHybridExecution", "omnibase_core.mixins.mixin_hybrid_execution"),
            ("MixinToolExecution", "omnibase_core.mixins.mixin_tool_execution"),
            ("MixinWorkflowSupport", "omnibase_core.mixins.mixin_workflow_support"),
            ("MixinHashComputation", "omnibase_core.mixins.mixin_hash_computation"),
            ("MixinCaching", "omnibase_core.mixins.mixin_caching"),
            ("MixinLazyEvaluation", "omnibase_core.mixins.mixin_lazy_evaluation"),
            ("MixinCompletionData", "omnibase_core.mixins.mixin_completion_data"),
            (
                "MixinCanonicalYAMLSerializer",
                "omnibase_core.mixins.mixin_canonical_serialization",
            ),
            ("MixinYAMLSerialization", "omnibase_core.mixins.mixin_yaml_serialization"),
            ("MixinSerializable", "omnibase_core.mixins.mixin_serializable"),
            ("MixinRedaction", "omnibase_core.mixins.mixin_redaction"),
            ("MixinContractMetadata", "omnibase_core.mixins.mixin_contract_metadata"),
            (
                "MixinContractStateReducer",
                "omnibase_core.mixins.mixin_contract_state_reducer",
            ),
            (
                "MixinIntrospectFromContract",
                "omnibase_core.mixins.mixin_introspect_from_contract",
            ),
            (
                "MixinNodeIdFromContract",
                "omnibase_core.mixins.mixin_node_id_from_contract",
            ),
            ("MixinNodeIntrospection", "omnibase_core.mixins.mixin_introspection"),
            ("MixinCLIHandler", "omnibase_core.mixins.mixin_cli_handler"),
            (
                "MixinDebugDiscoveryLogging",
                "omnibase_core.mixins.mixin_debug_discovery_logging",
            ),
            ("MixinFailFast", "omnibase_core.mixins.mixin_fail_fast"),
        ]

        for mixin_name, module_path in core_mixins:
            self._mixin_registry[mixin_name] = {
                "module": module_path,
                "dependencies": self._mixin_dependencies.get(mixin_name, []),
                "nodeeffect_overlap": "NONE",
                "import_path": f"{module_path}.{mixin_name}",
            }

        logger.debug(
            f"Loaded {len(self._mixin_registry)} mixins from fallback registry"
        )

    def parse_contract_file(self, contract_path: Path | str) -> ModelEnhancedContract:
        """
        Parse contract YAML file.

        Args:
            contract_path: Path to contract YAML file

        Returns:
            ModelEnhancedContract with parsed data and validation results

        Example:
            >>> parser = YAMLContractParser()
            >>> contract = parser.parse_contract_file("contracts/my_node.yaml")
            >>> if contract.is_valid:
            ...     print(f"Loaded: {contract.name} v{contract.version}")
        """
        contract_path = Path(contract_path)

        if not contract_path.exists():
            raise FileNotFoundError(f"Contract file not found: {contract_path}")

        try:
            with open(contract_path) as f:
                contract_data = yaml.safe_load(f)

            return self.parse_contract(contract_data)

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {contract_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load contract file {contract_path}: {e}")
            raise

    def parse_contract(self, contract_data: dict[str, Any]) -> ModelEnhancedContract:
        """
        Parse contract data dict.

        Args:
            contract_data: Contract data dictionary (from YAML)

        Returns:
            ModelEnhancedContract with parsed data and validation results

        Performs:
        1. Basic structure validation
        2. JSON Schema validation
        3. Mixin parsing and validation
        4. Advanced features parsing
        5. Dependency checking
        6. Backward compatibility handling

        Example:
            >>> contract_data = {
            ...     "name": "NodeMyEffect",
            ...     "version": {"major": 1, "minor": 0, "patch": 0},
            ...     "node_type": "effect",
            ...     "description": "My effect node",
            ...     "mixins": [{"name": "MixinHealthCheck", "enabled": True}]
            ... }
            >>> contract = parser.parse_contract(contract_data)
        """
        # Parse core fields
        # Support both 'name' and 'node_id' fields
        name = contract_data.get("name") or contract_data.get("node_id", "")
        version_data = contract_data.get("version", {})

        # Handle both string and dict version formats
        if isinstance(version_data, str):
            # Parse version string like "v1_0_0" or "1.0.0"
            version_str = version_data.lstrip("v").replace("_", ".")
            parts = version_str.split(".")
            major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 1
            minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        else:
            # Dict format
            major = version_data.get("major", 1)
            minor = version_data.get("minor", 0)
            patch = version_data.get("patch", 0)

        version = ModelVersionInfo(
            major=major,
            minor=minor,
            patch=patch,
        )
        node_type = contract_data.get("node_type", "")

        # Support both direct 'description' and metadata.description fields
        description = contract_data.get("description", "")
        if not description and "metadata" in contract_data:
            metadata = contract_data["metadata"]
            description = metadata.get("description", "")

        schema_version = contract_data.get("schema_version", "v1.0.0")

        # Create contract instance
        contract = ModelEnhancedContract(
            name=name,
            version=version,
            node_type=node_type,
            description=description,
            schema_version=schema_version,
            capabilities=contract_data.get("capabilities", []),
            endpoints=contract_data.get("endpoints", {}),
            dependencies=contract_data.get("dependencies", {}),
            configuration=contract_data.get("configuration", {}),
            subcontracts=contract_data.get("subcontracts", {}),
            performance_targets=contract_data.get("performance_targets", {}),
            health_checks=contract_data.get("health_checks", {}),
            input_state=contract_data.get("input_state", {}),
            output_state=contract_data.get("output_state", {}),
            io_operations=contract_data.get("io_operations", []),
            definitions=contract_data.get("definitions", {}),
            error_handling=contract_data.get("error_handling", {}),
        )

        # Validate required fields
        self._validate_required_fields(contract_data, contract)

        # Validate against JSON Schema (skip if using node_id instead of name)
        # Some older contracts use 'node_id' which isn't in the schema
        if "name" in contract_data:
            self._validate_contract_schema(contract_data, contract)

        # Parse mixins (v2.0 only)
        if "mixins" in contract_data:
            self._parse_mixins(contract_data["mixins"], contract)

        # Parse advanced_features (v2.0 only)
        if "advanced_features" in contract_data:
            self._parse_advanced_features(contract_data["advanced_features"], contract)

        # Parse Phase 3 fields (v2.1+)
        if "template" in contract_data:
            contract.template = self._parse_template_configuration(
                contract_data["template"], contract
            )

        if "generation" in contract_data:
            contract.generation = self._parse_generation_directives(
                contract_data["generation"], contract
            )

        if "quality_gates" in contract_data:
            contract.quality_gates = self._parse_quality_gates_configuration(
                contract_data["quality_gates"], contract
            )

        # Check for deprecated error_handling field
        if contract.has_deprecated_error_handling():
            contract.add_validation_error(
                "DEPRECATED: 'error_handling' field is deprecated. "
                "Use 'advanced_features' instead."
            )
            logger.warning(
                f"Contract {contract.name} uses deprecated 'error_handling' field"
            )

        # Validate mixin dependencies
        self._validate_mixin_dependencies(contract)

        # Log summary
        if contract.is_valid:
            logger.info(
                f"Parsed contract: {contract.name} v{contract.version} "
                f"({len(contract.mixins)} mixins, {contract.schema_version})"
            )
        else:
            logger.warning(
                f"Contract {contract.name} has {len(contract.validation_errors)} validation errors"
            )

        return contract

    def _validate_required_fields(
        self, contract_data: dict[str, Any], contract: ModelEnhancedContract
    ) -> None:
        """
        Validate that required fields are present in contract data.

        Args:
            contract_data: Raw contract data
            contract: Contract instance to update with errors
        """
        # Check for name field (can be either 'name' or 'node_id')
        if not contract_data.get("name") and not contract_data.get("node_id"):
            error_msg = "Required field 'name' is missing from contract"
            contract.add_validation_error(error_msg)
            logger.error(error_msg)

        # Check for version field
        if "version" not in contract_data:
            error_msg = "Required field 'version' is missing from contract"
            contract.add_validation_error(error_msg)
            logger.error(error_msg)
        elif isinstance(contract_data.get("version"), dict):
            # Validate version components if version is a dict
            version_data = contract_data["version"]
            if "major" not in version_data:
                error_msg = "Required field 'version.major' is missing from contract"
                contract.add_validation_error(error_msg)
                logger.error(error_msg)
            if "minor" not in version_data:
                error_msg = "Required field 'version.minor' is missing from contract"
                contract.add_validation_error(error_msg)
                logger.error(error_msg)
            if "patch" not in version_data:
                error_msg = "Required field 'version.patch' is missing from contract"
                contract.add_validation_error(error_msg)
                logger.error(error_msg)

        # Check for node_type field
        if not contract_data.get("node_type"):
            error_msg = "Required field 'node_type' is missing from contract"
            contract.add_validation_error(error_msg)
            logger.error(error_msg)

        # Check for description field
        if not contract_data.get("description"):
            # Also check metadata.description as fallback
            if "metadata" not in contract_data or not contract_data["metadata"].get(
                "description"
            ):
                error_msg = "Required field 'description' is missing from contract"
                contract.add_validation_error(error_msg)
                logger.error(error_msg)

    def _validate_contract_schema(
        self, contract_data: dict[str, Any], contract: ModelEnhancedContract
    ) -> None:
        """
        Validate contract against JSON Schema.

        Args:
            contract_data: Raw contract data
            contract: Contract instance to update with errors
        """
        try:
            # Validate against contract schema with resolver for $ref support
            validate(
                instance=contract_data,
                schema=self.contract_schema,
                resolver=self.schema_resolver,
            )
            logger.debug(f"Contract {contract.name} passed JSON Schema validation")

        except ValidationError as e:
            error_path = ".".join(str(p) for p in e.path) if e.path else "root"
            error_msg = f"JSON Schema validation failed at {error_path}: {e.message}"
            contract.add_validation_error(error_msg)
            logger.error(error_msg)

    def _parse_mixins(
        self, mixins_data: list[dict[str, Any]], contract: ModelEnhancedContract
    ) -> None:
        """
        Parse and validate mixin declarations.

        Args:
            mixins_data: List of mixin declarations from contract
            contract: Contract instance to update with parsed mixins
        """
        for mixin_data in mixins_data:
            mixin = self._parse_mixin_declaration(mixin_data, contract)
            contract.mixins.append(mixin)

    def _parse_mixin_declaration(
        self, mixin_data: dict[str, Any], contract: ModelEnhancedContract
    ) -> ModelMixinDeclaration:
        """
        Parse and validate single mixin declaration.

        Args:
            mixin_data: Mixin declaration data
            contract: Contract instance for context

        Returns:
            ModelMixinDeclaration with validation results
        """
        name = mixin_data.get("name", "")
        enabled = mixin_data.get("enabled", True)
        config = mixin_data.get("config", {})

        mixin = ModelMixinDeclaration(
            name=name,
            enabled=enabled,
            config=config,
        )

        # Validate mixin name pattern
        if not re.match(r"^Mixin[A-Z][a-zA-Z0-9]*$", name):
            error_msg = (
                f"Invalid mixin name '{name}': must follow Mixin* pattern "
                f"(e.g., MixinHealthCheck)"
            )
            mixin.add_validation_error(error_msg)
            contract.add_validation_error(error_msg)
            return mixin

        # Check if mixin exists in registry
        if name not in self._mixin_registry:
            error_msg = (
                f"Unknown mixin '{name}': not found in registry. "
                f"Available mixins: {', '.join(sorted(self._mixin_registry.keys()))}"
            )
            mixin.add_validation_error(error_msg)
            contract.add_validation_error(error_msg)
            return mixin

        # Get mixin metadata from registry
        mixin_meta = self._mixin_registry[name]
        mixin.import_path = mixin_meta["import_path"]

        # Validate mixin configuration against schema (if available)
        self._validate_mixin_config(name, config, mixin, contract)

        # Check for NodeEffect overlap
        overlap = mixin_meta.get("nodeeffect_overlap", "NONE")
        if overlap == "PARTIAL" and contract.node_type == "effect":
            logger.info(
                f"Mixin {name} has PARTIAL overlap with NodeEffect built-in features. "
                f"Ensure no feature duplication."
            )

        logger.debug(
            f"Parsed mixin: {name} (enabled={enabled}, config={len(config)} keys)"
        )

        return mixin

    def _validate_mixin_config(
        self,
        mixin_name: str,
        config: dict[str, Any],
        mixin: ModelMixinDeclaration,
        contract: ModelEnhancedContract,
    ) -> None:
        """
        Validate mixin configuration against JSON Schema.

        Args:
            mixin_name: Mixin name
            config: Configuration dict
            mixin: Mixin declaration to update with errors
            contract: Contract instance to update with errors
        """
        # Map mixin names to their schema definitions
        mixin_config_schemas = {
            "MixinHealthCheck": "MixinHealthCheckConfig",
            "MixinMetrics": "MixinMetricsConfig",
            "MixinEventDrivenNode": "MixinEventDrivenNodeConfig",
            "MixinServiceRegistry": "MixinServiceRegistryConfig",
            "MixinLogData": "MixinLogDataConfig",
        }

        schema_def_name = mixin_config_schemas.get(mixin_name)
        if not schema_def_name:
            # No specific schema defined for this mixin (generic config allowed)
            logger.debug(
                f"No JSON Schema validation for {mixin_name} config (generic allowed)"
            )
            return

        # Get schema definition from mixins_schema
        if "definitions" not in self.mixins_schema:
            logger.warning("Mixins schema missing 'definitions' section")
            return

        schema_def = self.mixins_schema["definitions"].get(schema_def_name)
        if not schema_def:
            logger.warning(
                f"Schema definition '{schema_def_name}' not found in mixins_schema"
            )
            return

        # Validate config against schema
        try:
            validate(instance=config, schema=schema_def)
            logger.debug(f"Mixin {mixin_name} config passed validation")

        except ValidationError as e:
            error_path = ".".join(str(p) for p in e.path) if e.path else "root"
            error_msg = f"Mixin {mixin_name} config validation failed at {error_path}: {e.message}"
            mixin.add_validation_error(error_msg)
            contract.add_validation_error(error_msg)
            logger.error(error_msg)

    def _parse_advanced_features(
        self, features_data: dict[str, Any], contract: ModelEnhancedContract
    ) -> None:
        """
        Parse advanced_features section.

        Args:
            features_data: Advanced features data from contract
            contract: Contract instance to update
        """
        # Parse circuit_breaker
        circuit_breaker = None
        if "circuit_breaker" in features_data:
            cb_data = features_data["circuit_breaker"]
            circuit_breaker = ModelCircuitBreakerConfig(
                enabled=cb_data.get("enabled", True),
                failure_threshold=cb_data.get("failure_threshold", 5),
                recovery_timeout_ms=cb_data.get("recovery_timeout_ms", 60000),
                half_open_max_calls=cb_data.get("half_open_max_calls", 3),
                services=cb_data.get("services", []),
            )

        # Parse retry_policy
        retry_policy = None
        if "retry_policy" in features_data:
            rp_data = features_data["retry_policy"]
            retry_policy = ModelRetryPolicyConfig(
                enabled=rp_data.get("enabled", True),
                max_attempts=rp_data.get("max_attempts", 3),
                initial_delay_ms=rp_data.get("initial_delay_ms", 1000),
                max_delay_ms=rp_data.get("max_delay_ms", 30000),
                backoff_multiplier=rp_data.get("backoff_multiplier", 2.0),
                retryable_exceptions=rp_data.get(
                    "retryable_exceptions",
                    ["TimeoutError", "ConnectionError", "TemporaryFailure"],
                ),
                retryable_status_codes=rp_data.get(
                    "retryable_status_codes", [429, 500, 502, 503, 504]
                ),
            )

        # Parse dead_letter_queue
        dlq = None
        if "dead_letter_queue" in features_data:
            dlq_data = features_data["dead_letter_queue"]
            dlq = ModelDeadLetterQueueConfig(
                enabled=dlq_data.get("enabled", True),
                max_retries=dlq_data.get("max_retries", 3),
                topic_suffix=dlq_data.get("topic_suffix", ".dlq"),
                retry_delay_ms=dlq_data.get("retry_delay_ms", 5000),
                alert_threshold=dlq_data.get("alert_threshold", 100),
                monitoring=dlq_data.get("monitoring", {}),
            )

        # Parse transactions
        transactions = None
        if "transactions" in features_data:
            tx_data = features_data["transactions"]
            transactions = ModelTransactionsConfig(
                enabled=tx_data.get("enabled", True),
                isolation_level=tx_data.get("isolation_level", "READ_COMMITTED"),
                timeout_seconds=tx_data.get("timeout_seconds", 30),
                rollback_on_error=tx_data.get("rollback_on_error", True),
                savepoints=tx_data.get("savepoints", True),
            )

        # Parse security_validation
        security = None
        if "security_validation" in features_data:
            sec_data = features_data["security_validation"]
            security = ModelSecurityValidationConfig(
                enabled=sec_data.get("enabled", True),
                sanitize_inputs=sec_data.get("sanitize_inputs", True),
                sanitize_logs=sec_data.get("sanitize_logs", True),
                validate_sql=sec_data.get("validate_sql", True),
                max_input_length=sec_data.get("max_input_length", 10000),
                forbidden_patterns=sec_data.get(
                    "forbidden_patterns",
                    [
                        r"(?i)(DROP|DELETE|TRUNCATE)\s+TABLE",
                        r"(?i)EXEC(UTE)?\s+",
                    ],
                ),
                redact_fields=sec_data.get(
                    "redact_fields", ["password", "api_key", "secret", "token"]
                ),
            )

        # Parse observability
        observability = None
        if "observability" in features_data:
            obs_data = features_data["observability"]
            observability = ModelObservabilityConfig(
                tracing=obs_data.get("tracing", {}),
                metrics=obs_data.get("metrics", {}),
                logging=obs_data.get("logging", {}),
            )

        # Create advanced features instance
        contract.advanced_features = ModelAdvancedFeatures(
            circuit_breaker=circuit_breaker,
            retry_policy=retry_policy,
            dead_letter_queue=dlq,
            transactions=transactions,
            security_validation=security,
            observability=observability,
        )

        logger.debug(f"Parsed advanced_features for contract {contract.name}")

    def _parse_template_configuration(
        self, template_data: dict[str, Any], contract: ModelEnhancedContract
    ) -> ModelTemplateConfiguration:
        """
        Parse template configuration section (Phase 3).

        Args:
            template_data: Template configuration data from contract
            contract: Contract instance for error reporting

        Returns:
            ModelTemplateConfiguration instance
        """
        # Parse variant (default to STANDARD)
        variant_str = template_data.get("variant", "standard")
        try:
            variant = EnumTemplateVariant(variant_str)
        except ValueError:
            error_msg = (
                f"Invalid template variant '{variant_str}'. "
                f"Must be one of: {', '.join(v.value for v in EnumTemplateVariant)}"
            )
            contract.add_validation_error(error_msg)
            logger.error(error_msg)
            variant = EnumTemplateVariant.STANDARD

        # Parse custom_template (optional)
        custom_template = template_data.get("custom_template")
        if variant == EnumTemplateVariant.CUSTOM and not custom_template:
            error_msg = "Template variant 'custom' requires 'custom_template' path"
            contract.add_validation_error(error_msg)
            logger.error(error_msg)

        # Parse patterns
        patterns = template_data.get("patterns", [])
        if not isinstance(patterns, list):
            error_msg = (
                f"Template 'patterns' must be a list, got {type(patterns).__name__}"
            )
            contract.add_validation_error(error_msg)
            logger.error(error_msg)
            patterns = []

        # Parse pattern_configuration
        pattern_configuration = template_data.get("pattern_configuration", {})
        if not isinstance(pattern_configuration, dict):
            error_msg = (
                f"Template 'pattern_configuration' must be a dict, "
                f"got {type(pattern_configuration).__name__}"
            )
            contract.add_validation_error(error_msg)
            logger.error(error_msg)
            pattern_configuration = {}

        config = ModelTemplateConfiguration(
            variant=variant,
            custom_template=custom_template,
            patterns=patterns,
            pattern_configuration=pattern_configuration,
        )

        logger.debug(
            f"Parsed template configuration for {contract.name}: "
            f"variant={variant.value}, patterns={len(patterns)}"
        )

        return config

    def _parse_generation_directives(
        self, generation_data: dict[str, Any], contract: ModelEnhancedContract
    ) -> ModelGenerationDirectives:
        """
        Parse generation directives section (Phase 3).

        Args:
            generation_data: Generation directives data from contract
            contract: Contract instance for error reporting

        Returns:
            ModelGenerationDirectives instance
        """
        # Parse enable_llm
        enable_llm = generation_data.get("enable_llm", True)

        # Parse llm_tier
        llm_tier_str = generation_data.get("llm_tier", "CLOUD_FAST")
        try:
            llm_tier = EnumLLMTier(llm_tier_str)
        except ValueError:
            error_msg = (
                f"Invalid LLM tier '{llm_tier_str}'. "
                f"Must be one of: {', '.join(t.value for t in EnumLLMTier)}"
            )
            contract.add_validation_error(error_msg)
            logger.error(error_msg)
            llm_tier = EnumLLMTier.CLOUD_FAST

        # Parse quality_level
        quality_level_str = generation_data.get("quality_level", "standard")
        try:
            quality_level = EnumQualityLevel(quality_level_str)
        except ValueError:
            error_msg = (
                f"Invalid quality level '{quality_level_str}'. "
                f"Must be one of: {', '.join(q.value for q in EnumQualityLevel)}"
            )
            contract.add_validation_error(error_msg)
            logger.error(error_msg)
            quality_level = EnumQualityLevel.STANDARD

        # Parse fallback_strategy
        fallback_str = generation_data.get("fallback_strategy", "graceful")
        try:
            fallback_strategy = EnumFallbackStrategy(fallback_str)
        except ValueError:
            error_msg = (
                f"Invalid fallback strategy '{fallback_str}'. "
                f"Must be one of: {', '.join(f.value for f in EnumFallbackStrategy)}"
            )
            contract.add_validation_error(error_msg)
            logger.error(error_msg)
            fallback_strategy = EnumFallbackStrategy.GRACEFUL

        # Parse context enhancement flags
        include_patterns = generation_data.get("include_patterns", True)
        include_references = generation_data.get("include_references", True)
        max_context_size = generation_data.get("max_context_size", 8000)

        # Parse performance tuning
        timeout_seconds = generation_data.get("timeout_seconds", 30)
        retry_attempts = generation_data.get("retry_attempts", 3)

        directives = ModelGenerationDirectives(
            enable_llm=enable_llm,
            llm_tier=llm_tier,
            quality_level=quality_level,
            fallback_strategy=fallback_strategy,
            include_patterns=include_patterns,
            include_references=include_references,
            max_context_size=max_context_size,
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
        )

        logger.debug(
            f"Parsed generation directives for {contract.name}: "
            f"llm_tier={llm_tier.value}, quality={quality_level.value}"
        )

        return directives

    def _parse_quality_gates_configuration(
        self,
        quality_gates_data: dict[str, Any] | list[Any],
        contract: ModelEnhancedContract,
    ) -> ModelQualityGatesConfiguration:
        """
        Parse quality gates configuration section (Phase 3).

        Args:
            quality_gates_data: Quality gates data from contract (dict or list)
            contract: Contract instance for error reporting

        Returns:
            ModelQualityGatesConfiguration instance
        """
        # Handle legacy format where quality_gates is a list directly
        if isinstance(quality_gates_data, list):
            gates_list = quality_gates_data
            fail_on_first_error = True  # Default
            collect_warnings = True  # Default
        else:
            # Parse gates list
            gates_list = quality_gates_data.get("gates", [])
            # Parse configuration flags
            fail_on_first_error = quality_gates_data.get("fail_on_first_error", True)
            collect_warnings = quality_gates_data.get("collect_warnings", True)

        if not isinstance(gates_list, list):
            error_msg = (
                f"Quality gates 'gates' must be a list, got {type(gates_list).__name__}"
            )
            contract.add_validation_error(error_msg)
            logger.error(error_msg)
            gates_list = []

        gates = []
        for gate_data in gates_list:
            if not isinstance(gate_data, dict):
                error_msg = (
                    f"Quality gate must be a dict, got {type(gate_data).__name__}"
                )
                contract.add_validation_error(error_msg)
                logger.error(error_msg)
                continue

            # Parse gate fields
            gate_name = gate_data.get("gate", gate_data.get("name", ""))
            if not gate_name:
                error_msg = "Quality gate missing 'gate' or 'name' field"
                contract.add_validation_error(error_msg)
                logger.error(error_msg)
                continue

            required = gate_data.get("required", True)
            config = gate_data.get("config", {})
            description = gate_data.get("description", "")

            gate = ModelQualityGate(
                name=gate_name,
                required=required,
                config=config,
                description=description,
            )
            gates.append(gate)

        config = ModelQualityGatesConfiguration(
            gates=gates,
            fail_on_first_error=fail_on_first_error,
            collect_warnings=collect_warnings,
        )

        logger.debug(
            f"Parsed quality gates for {contract.name}: {len(gates)} gates "
            f"({len([g for g in gates if g.required])} required)"
        )

        return config

    def _validate_mixin_dependencies(self, contract: ModelEnhancedContract) -> None:
        """
        Validate that all mixin dependencies are satisfied.

        Args:
            contract: Contract to validate
        """
        enabled_mixins = set(contract.get_mixin_names())

        for mixin in contract.get_enabled_mixins():
            required_deps = self._mixin_dependencies.get(mixin.name, [])

            for dep in required_deps:
                if dep not in enabled_mixins:
                    error_msg = (
                        f"Mixin dependency not satisfied: {mixin.name} requires {dep}, "
                        f"but {dep} is not enabled in contract"
                    )
                    mixin.add_validation_error(error_msg)
                    contract.add_validation_error(error_msg)
                    logger.error(error_msg)

    def get_available_mixins(self) -> list[str]:
        """
        Get list of available mixin names from registry.

        Returns:
            Sorted list of mixin names

        Example:
            >>> parser = YAMLContractParser()
            >>> mixins = parser.get_available_mixins()
            >>> print(f"Available: {len(mixins)} mixins")
        """
        return sorted(self._mixin_registry.keys())

    def get_mixin_info(self, mixin_name: str) -> dict[str, Any] | None:
        """
        Get metadata for specific mixin.

        Args:
            mixin_name: Mixin name

        Returns:
            Mixin metadata dict or None if not found

        Example:
            >>> parser = YAMLContractParser()
            >>> info = parser.get_mixin_info("MixinHealthCheck")
            >>> print(info["import_path"])
            omnibase_core.mixins.mixin_health_check.MixinHealthCheck
        """
        return self._mixin_registry.get(mixin_name)

    def get_mixin_dependencies(self, mixin_name: str) -> list[str]:
        """
        Get dependencies for specific mixin.

        Args:
            mixin_name: Mixin name

        Returns:
            List of required mixin names (empty if no dependencies)

        Example:
            >>> parser = YAMLContractParser()
            >>> deps = parser.get_mixin_dependencies("MixinEventDrivenNode")
            >>> print(deps)
            ['MixinEventHandler', 'MixinNodeLifecycle', 'MixinIntrospectionPublisher']
        """
        return self._mixin_dependencies.get(mixin_name, [])


# Export
__all__ = ["YAMLContractParser"]
