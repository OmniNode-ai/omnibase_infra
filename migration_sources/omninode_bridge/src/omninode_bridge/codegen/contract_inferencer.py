#!/usr/bin/env python3
"""
Contract Inferencer for automated ONEX v2.0 contract generation.

Analyzes existing node.py files and generates ONEX v2.0 contracts automatically
using AST parsing and LLM-based configuration inference.

Pipeline:
1. Parse node.py to extract class structure, mixins, and methods
2. Detect mixins from imports and inheritance chain
3. Use LLM to infer mixin configurations based on business logic
4. Generate v2.0 contract with proper schema and configurations
5. Validate and serialize to YAML

ONEX v2.0 Compliance:
- Async/await throughout
- ModelOnexError for error handling
- Structured logging with emit_log_event
- Comprehensive validation

Example:
    >>> import os
    >>> os.environ["ZAI_API_KEY"] = "your_api_key"  # pragma: allowlist secret
    >>> inferencer = ContractInferencer(enable_llm=True)
    >>> contract_yaml = await inferencer.infer_from_node(
    ...     node_path="path/to/node.py"
    ... )
    >>> print(contract_yaml)
"""

import ast
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.contracts.model_io_operation_config import (
    ModelIOOperationConfig,
)
from omnibase_core.models.core import ModelContainer
from omnibase_core.primitives.model_semver import ModelSemVer

from omninode_bridge.codegen.mixin_injector import MIXIN_CATALOG
from omninode_bridge.codegen.models_contract import (
    ModelAdvancedFeatures,
    ModelCircuitBreakerConfig,
    ModelDeadLetterQueueConfig,
    ModelEnhancedContract,
    ModelMixinDeclaration,
    ModelObservabilityConfig,
    ModelRetryPolicyConfig,
    ModelSecurityValidationConfig,
    ModelTransactionsConfig,
    ModelVersionInfo,
)
from omninode_bridge.nodes.llm_effect.v1_0_0.models.enum_llm_tier import EnumLLMTier
from omninode_bridge.nodes.llm_effect.v1_0_0.models.model_request import ModelLLMRequest
from omninode_bridge.nodes.llm_effect.v1_0_0.node import NodeLLMEffect

logger = logging.getLogger(__name__)


@dataclass
class ModelNodeAnalysis:
    """
    Analysis results from parsing a node.py file.

    Contains all information extracted from AST parsing needed to generate
    a contract.

    Attributes:
        node_name: Node class name (e.g., NodeLLMEffect)
        node_type: Node type (effect, compute, reducer, orchestrator)
        base_class: Direct parent class (e.g., NodeEffect)
        mixins_detected: List of mixin names found in inheritance
        imports: Dict of import statements {module: [names]}
        methods: List of method names defined in class
        docstring: Class-level docstring
        version: Detected version from filename/docstring
        io_operations: Detected I/O operation types (API, database, etc.)
        file_path: Original file path
    """

    node_name: str
    node_type: str
    base_class: str
    mixins_detected: list[str]
    imports: dict[str, list[str]]
    methods: list[str]
    docstring: Optional[str]
    version: str
    io_operations: list[str]
    file_path: str


@dataclass
class ModelMixinConfigInference:
    """
    LLM-inferred mixin configuration.

    Attributes:
        mixin_name: Mixin class name
        confidence: Confidence score (0.0-1.0)
        inferred_config: Inferred configuration dict
        reasoning: LLM's reasoning for the configuration
    """

    mixin_name: str
    confidence: float
    inferred_config: dict[str, Any]
    reasoning: str


class ContractInferencer:
    """
    Infer ONEX v2.0 contracts from existing node implementations.

    Analyzes Python node.py files using AST parsing and LLM inference to
    automatically generate complete contracts with proper mixin configurations.

    Thread-safe for analysis (AST parsing). LLM calls are async.

    Example:
        >>> inferencer = ContractInferencer(enable_llm=True)
        >>> contract_yaml = await inferencer.infer_from_node("path/to/node.py")
        >>> with open("contract.yaml", "w") as f:
        ...     f.write(contract_yaml)
    """

    def __init__(
        self,
        enable_llm: bool = True,
        llm_tier: EnumLLMTier = EnumLLMTier.CLOUD_FAST,
    ):
        """
        Initialize contract inferencer.

        Args:
            enable_llm: Enable LLM for config inference (if False, uses defaults)
            llm_tier: LLM tier to use for inference

        Raises:
            ModelOnexError: If ZAI_API_KEY not set when enable_llm=True
        """
        self.enable_llm = enable_llm
        self.llm_tier = llm_tier

        # Initialize NodeLLMEffect if LLM enabled
        self.llm_node: Optional[NodeLLMEffect]
        if self.enable_llm:
            zai_api_key = os.getenv("ZAI_API_KEY")
            if not zai_api_key:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.INVALID_INPUT,
                    message="ZAI_API_KEY environment variable required when enable_llm=True",
                    details={"enable_llm": enable_llm},
                )

            container = ModelContainer(value={}, container_type="config")
            self.llm_node = NodeLLMEffect(container)
        else:
            self.llm_node = None

        logger.info(
            f"ContractInferencer initialized (enable_llm={enable_llm}, tier={llm_tier.value})"
        )

    async def infer_from_node(
        self,
        node_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> str:
        """
        Infer contract from node.py file.

        Args:
            node_path: Path to node.py file
            output_path: Optional path to write contract.yaml (if provided)

        Returns:
            Contract YAML as string

        Raises:
            ModelOnexError: On parsing or inference failures
        """
        try:
            # Step 1: Parse node file
            logger.info(f"Analyzing node file: {node_path}")
            analysis = self._parse_node_file(node_path)

            # Step 2: Infer mixin configurations
            logger.info(f"Detected {len(analysis.mixins_detected)} mixins")
            mixin_configs = await self._infer_mixin_configs(analysis)

            # Step 3: Generate enhanced contract
            logger.info("Generating contract from analysis")
            contract = self._generate_contract(analysis, mixin_configs)

            # Step 4: Serialize to YAML
            contract_yaml = self._serialize_to_yaml(contract)

            # Step 5: Write to file if requested
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(contract_yaml)
                logger.info(f"Contract written to {output_path}")

            return contract_yaml

        except Exception as e:
            logger.error(f"Failed to infer contract from {node_path}: {e}")
            if isinstance(e, ModelOnexError):
                raise
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"Failed to infer contract: {e}",
                details={"node_path": str(node_path), "error_type": type(e).__name__},
            )

    def _parse_node_file(self, node_path: str | Path) -> ModelNodeAnalysis:
        """
        Parse node.py file using AST.

        Extracts:
        - Class name and inheritance
        - Mixin imports
        - Method signatures
        - Docstrings
        - Version information
        - I/O operations

        Args:
            node_path: Path to node.py file

        Returns:
            ModelNodeAnalysis with extracted information

        Raises:
            ModelOnexError: On parsing errors
        """
        try:
            node_path = Path(node_path)
            source_code = node_path.read_text()

            # Parse AST
            tree = ast.parse(source_code)

            # Extract imports
            imports: dict[str, list[str]] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports[node.module] = [
                            alias.name for alias in node.names if alias.name
                        ]
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports[alias.name] = [alias.name]

            # Find the main node class
            node_class = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Look for classes that inherit from NodeEffect, NodeCompute, etc.
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id.startswith("Node"):
                            node_class = node
                            break
                    if node_class:
                        break

            if not node_class:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.INVALID_INPUT,
                    message="No node class found in file",
                    details={"file": str(node_path)},
                )

            # Extract class information
            node_name = node_class.name
            docstring = ast.get_docstring(node_class)

            # Extract base class and mixins
            base_class = ""
            mixins_detected = []
            for base in node_class.bases:
                if isinstance(base, ast.Name):
                    base_name = base.id
                    if base_name.startswith("Node"):
                        base_class = base_name
                    elif base_name.startswith("Mixin"):
                        mixins_detected.append(base_name)

            # Detect node type from base class
            node_type = ""
            if "Effect" in base_class:
                node_type = "effect"
            elif "Compute" in base_class:
                node_type = "compute"
            elif "Reducer" in base_class:
                node_type = "reducer"
            elif "Orchestrator" in base_class:
                node_type = "orchestrator"

            # Extract methods
            methods = [
                item.name
                for item in node_class.body
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
            ]

            # Detect version from path (e.g., v1_0_0)
            version = "1.0.0"
            version_match = node_path.parts
            for part in version_match:
                if part.startswith("v") and "_" in part:
                    # Convert v1_0_0 to 1.0.0
                    version = part[1:].replace("_", ".")
                    break

            # Detect I/O operations from code analysis
            io_operations = self._detect_io_operations(source_code, imports)

            return ModelNodeAnalysis(
                node_name=node_name,
                node_type=node_type,
                base_class=base_class,
                mixins_detected=mixins_detected,
                imports=imports,
                methods=methods,
                docstring=docstring,
                version=version,
                io_operations=io_operations,
                file_path=str(node_path),
            )

        except SyntaxError as e:
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                message=f"Failed to parse node file: {e}",
                details={"file": str(node_path)},
            )

    def _detect_io_operations(
        self, source_code: str, imports: dict[str, list[str]]
    ) -> list[str]:
        """
        Detect I/O operation types from source code.

        Looks for:
        - HTTP clients (httpx, requests)
        - Database clients (asyncpg, sqlalchemy)
        - Message queues (aiokafka, redis)
        - File operations (open, Path)
        - External APIs (specific patterns)

        Args:
            source_code: Source code string
            imports: Detected imports

        Returns:
            List of I/O operation types
        """
        operations = []

        # HTTP operations
        if any(
            module in imports for module in ["httpx", "requests", "aiohttp", "urllib"]
        ):
            operations.append("http_request")

        # Database operations
        if any(
            module in imports
            for module in ["asyncpg", "sqlalchemy", "psycopg2", "pymongo"]
        ):
            operations.append("database_query")

        # Message queue operations
        if any(module in imports for module in ["aiokafka", "redis", "pika"]):
            operations.append("message_queue")

        # File operations
        if "open(" in source_code or "Path(" in source_code:
            operations.append("file_io")

        # Default if no specific operations detected
        if not operations:
            operations.append("computation")

        return operations

    async def _infer_mixin_configs(
        self, analysis: ModelNodeAnalysis
    ) -> list[ModelMixinConfigInference]:
        """
        Infer mixin configurations using LLM.

        For each detected mixin, asks LLM to infer appropriate configuration
        based on business logic patterns in the node.

        Args:
            analysis: Node analysis results

        Returns:
            List of inferred mixin configurations

        Raises:
            ModelOnexError: On LLM inference failures
        """
        if not self.enable_llm:
            # Return default configs if LLM disabled
            configs = []
            for mixin in analysis.mixins_detected:
                catalog_entry = MIXIN_CATALOG.get(mixin, {})
                default_config = catalog_entry.get("default_config")
                if default_config is None:
                    raise ModelOnexError(
                        error_code=EnumCoreErrorCode.INVALID_STATE,
                        message=f"No fallback configuration for {mixin}",
                        details={"mixin": mixin, "catalog_entry": catalog_entry},
                    )
                configs.append(
                    ModelMixinConfigInference(
                        mixin_name=mixin,
                        confidence=0.5,
                        inferred_config=default_config,
                        reasoning="LLM disabled - using catalog defaults",
                    )
                )
            return configs

        inferred_configs = []

        for mixin_name in analysis.mixins_detected:
            try:
                # Build prompt for this mixin
                prompt = self._build_mixin_config_prompt(analysis, mixin_name)

                # Call LLM
                config = await self._call_llm_for_config(prompt, mixin_name)
                inferred_configs.append(config)

            except Exception as e:
                logger.warning(f"Failed to infer config for {mixin_name}: {e}")
                # Fallback to catalog default config
                catalog_entry = MIXIN_CATALOG.get(mixin_name, {})
                default_config = catalog_entry.get("default_config")
                if default_config is None:
                    raise ModelOnexError(
                        error_code=EnumCoreErrorCode.INVALID_STATE,
                        message=f"No fallback configuration for {mixin_name}",
                        details={
                            "mixin": mixin_name,
                            "catalog_entry": catalog_entry,
                            "original_error": str(e),
                        },
                    )
                inferred_configs.append(
                    ModelMixinConfigInference(
                        mixin_name=mixin_name,
                        confidence=0.0,
                        inferred_config=default_config,
                        reasoning=f"Inference failed: {e} - using catalog defaults",
                    )
                )

        return inferred_configs

    def _build_mixin_config_prompt(
        self, analysis: ModelNodeAnalysis, mixin_name: str
    ) -> str:
        """
        Build LLM prompt for mixin configuration inference.

        Args:
            analysis: Node analysis results
            mixin_name: Name of mixin to infer config for

        Returns:
            Formatted prompt string
        """
        # Get mixin metadata from catalog
        mixin_info = MIXIN_CATALOG.get(mixin_name, {})
        mixin_description = mixin_info.get("description", "No description available")

        prompt_parts = [
            f"# Task: Infer configuration for {mixin_name}",
            "",
            "## Mixin Information",
            f"Name: {mixin_name}",
            f"Description: {mixin_description}",
            "",
            "## Node Context",
            f"Node Name: {analysis.node_name}",
            f"Node Type: {analysis.node_type}",
            f"Purpose: {analysis.docstring or 'No description'}",
            f"I/O Operations: {', '.join(analysis.io_operations)}",
            f"Methods: {', '.join(analysis.methods[:10])}",  # First 10 methods
            "",
            "## Configuration Examples",
        ]

        # Add mixin-specific examples
        if mixin_name == "MixinMetrics":
            prompt_parts.extend(
                [
                    "Example configuration:",
                    "```yaml",
                    "collect_latency: true",
                    "collect_throughput: true",
                    "collect_error_rates: true",
                    "percentiles: [50, 95, 99]",
                    "histogram_buckets: [100, 500, 1000, 2000, 5000]",
                    "```",
                ]
            )
        elif mixin_name == "MixinHealthCheck":
            prompt_parts.extend(
                [
                    "Example configuration:",
                    "```yaml",
                    "check_interval_ms: 60000",
                    "timeout_seconds: 10.0",
                    "components:",
                    "  - name: external_api",
                    "    critical: true",
                    "    timeout_seconds: 5.0",
                    "```",
                ]
            )
        elif mixin_name == "MixinCaching":
            prompt_parts.extend(
                [
                    "Example configuration:",
                    "```yaml",
                    "cache_ttl_seconds: 300",
                    "max_cache_size: 1000",
                    "eviction_policy: lru",
                    "```",
                ]
            )

        prompt_parts.extend(
            [
                "",
                "## Requirements",
                "1. Return ONLY a valid YAML configuration block (no markdown fences)",
                "2. Include appropriate configuration values based on node's purpose",
                "3. Consider I/O operations when setting timeouts/thresholds",
                "4. Use reasonable defaults for standard deployments",
                "5. Include a 'confidence' field (0.0-1.0) indicating your confidence",
                "6. Include a 'reasoning' field explaining your choices",
                "",
                "Generate the configuration:",
            ]
        )

        return "\n".join(prompt_parts)

    async def _call_llm_for_config(
        self, prompt: str, mixin_name: str
    ) -> ModelMixinConfigInference:
        """
        Call LLM to infer mixin configuration.

        Args:
            prompt: Formatted prompt
            mixin_name: Mixin name

        Returns:
            ModelMixinConfigInference with inferred config

        Raises:
            ModelOnexError: On LLM call failures
        """
        if self.llm_node is None:
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.INVALID_STATE,
                message="LLM node not initialized",
                details={"enable_llm": self.enable_llm},
            )

        # Initialize LLM node if not already done
        if self.llm_node.http_client is None:
            await self.llm_node.initialize()

        # Build LLM request
        llm_request = ModelLLMRequest(
            prompt=prompt,
            tier=self.llm_tier,
            max_tokens=1000,
            temperature=0.3,  # Lower temperature for more consistent configs
            top_p=0.9,
            system_prompt="You are an expert in ONEX v2.0 mixin configuration. Generate precise, production-ready configurations.",
            operation_type="mixin_config_inference",
        )

        # Create contract for LLM call
        from omnibase_core.enums.enum_node_type import EnumNodeType

        contract = ModelContractEffect(
            name="mixin_config_inference",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description=f"Infer configuration for {mixin_name}",
            node_type=EnumNodeType.EFFECT,
            input_model="ModelLLMRequest",
            output_model="ModelLLMResponse",
            input_state=llm_request.model_dump(),
            io_operations=[
                ModelIOOperationConfig(
                    operation_type="llm_request",
                    atomic=True,
                    timeout_seconds=30,
                )
            ],
        )

        # Call LLM
        llm_response = await self.llm_node.execute_effect(contract)
        generated_text = llm_response.generated_text.strip()

        # Parse YAML response
        try:
            # Strip markdown code fences if present
            if generated_text.startswith("```yaml"):
                generated_text = generated_text[len("```yaml") :].strip()
            if generated_text.startswith("```"):
                generated_text = generated_text[3:].strip()
            if generated_text.endswith("```"):
                generated_text = generated_text[:-3].strip()

            config_data = yaml.safe_load(generated_text)

            # Extract confidence and reasoning
            confidence = float(config_data.pop("confidence", 0.7))
            reasoning = config_data.pop("reasoning", "LLM-inferred configuration")

            return ModelMixinConfigInference(
                mixin_name=mixin_name,
                confidence=confidence,
                inferred_config=config_data,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.warning(f"Failed to parse LLM response for {mixin_name}: {e}")
            return ModelMixinConfigInference(
                mixin_name=mixin_name,
                confidence=0.0,
                inferred_config={},
                reasoning=f"Failed to parse LLM response: {e}",
            )

    def _generate_contract(
        self,
        analysis: ModelNodeAnalysis,
        mixin_configs: list[ModelMixinConfigInference],
    ) -> ModelEnhancedContract:
        """
        Generate ModelEnhancedContract from analysis and inferred configs.

        Args:
            analysis: Node analysis results
            mixin_configs: Inferred mixin configurations

        Returns:
            ModelEnhancedContract ready for serialization
        """
        # Parse version
        version_parts = analysis.version.split(".")
        version = ModelVersionInfo(
            major=int(version_parts[0]),
            minor=int(version_parts[1]) if len(version_parts) > 1 else 0,
            patch=int(version_parts[2]) if len(version_parts) > 2 else 0,
        )

        # Build mixin declarations
        mixins = [
            ModelMixinDeclaration(
                name=config.mixin_name,
                enabled=True,
                config=config.inferred_config,
                import_path=MIXIN_CATALOG.get(config.mixin_name, {}).get(
                    "import_path", ""
                ),
            )
            for config in mixin_configs
        ]

        # Build advanced features (defaults for all nodes)
        advanced_features = ModelAdvancedFeatures(
            circuit_breaker=ModelCircuitBreakerConfig(
                enabled=True,
                failure_threshold=5,
                recovery_timeout_ms=60000,
                half_open_max_calls=3,
            ),
            retry_policy=ModelRetryPolicyConfig(
                enabled=True,
                max_attempts=3,
                initial_delay_ms=1000,
                max_delay_ms=10000,
                backoff_multiplier=2.0,
            ),
            observability=ModelObservabilityConfig(
                tracing={"enabled": True, "sample_rate": 1.0},
                metrics={"enabled": True, "export_interval_seconds": 15},
                logging={
                    "structured": True,
                    "json_format": True,
                    "correlation_tracking": True,
                },
            ),
        )

        # Add DLQ for effect nodes
        if analysis.node_type == "effect":
            advanced_features.dead_letter_queue = ModelDeadLetterQueueConfig(
                enabled=True,
                max_retries=3,
                topic_suffix=".dlq",
                retry_delay_ms=5000,
                alert_threshold=100,
            )

        # Add transactions for database operations
        if "database_query" in analysis.io_operations:
            advanced_features.transactions = ModelTransactionsConfig(
                enabled=True,
                isolation_level="READ_COMMITTED",
                timeout_seconds=30,
                rollback_on_error=True,
            )

        # Add security validation
        advanced_features.security_validation = ModelSecurityValidationConfig(
            enabled=True,
            sanitize_inputs=True,
            sanitize_logs=True,
            validate_sql="database_query" in analysis.io_operations,
        )

        # Extract node name (remove "Node" prefix)
        name = analysis.node_name
        if name.startswith("Node"):
            name = name[4:]  # Remove "Node" prefix

        # Convert camelCase to snake_case (improved logic)
        # Handle consecutive uppercase letters (e.g., LLM -> llm, not l_l_m)
        import re

        # Insert underscore before uppercase letter that's followed by lowercase
        name = re.sub("([a-z])([A-Z])", r"\1_\2", name)
        # Insert underscore before uppercase letter that follows a lowercase or digit
        name = re.sub("([a-zA-Z])([A-Z][a-z])", r"\1_\2", name)
        name = name.lower()

        # Build capabilities from I/O operations and mixins
        capabilities = []
        for op in analysis.io_operations:
            capabilities.append(
                {"name": op, "description": f"{op.replace('_', ' ').title()} support"}
            )
        for mixin_config in mixin_configs:
            mixin_info = MIXIN_CATALOG.get(mixin_config.mixin_name, {})
            capabilities.append(
                {
                    "name": mixin_config.mixin_name.lower().replace("mixin", ""),
                    "description": mixin_info.get("description", ""),
                }
            )

        return ModelEnhancedContract(
            schema_version="v2.0.0",
            name=name,
            version=version,
            node_type=analysis.node_type,
            description=analysis.docstring or f"{name} node implementation",
            capabilities=capabilities,
            mixins=mixins,
            advanced_features=advanced_features,
        )

    def _serialize_to_yaml(self, contract: ModelEnhancedContract) -> str:
        """
        Serialize ModelEnhancedContract to YAML string.

        Args:
            contract: Enhanced contract to serialize

        Returns:
            YAML string with proper formatting
        """
        # Build contract dict
        contract_dict = {
            "schema_version": contract.schema_version,
            "name": contract.name,
            "version": {
                "major": contract.version.major,
                "minor": contract.version.minor,
                "patch": contract.version.patch,
            },
            "node_type": contract.node_type,
            "description": contract.description,
        }

        # Add capabilities
        if contract.capabilities:
            contract_dict["capabilities"] = contract.capabilities

        # Add mixins
        if contract.mixins:
            contract_dict["mixins"] = [
                {
                    "name": mixin.name,
                    "enabled": mixin.enabled,
                    "config": mixin.config,
                }
                for mixin in contract.mixins
            ]

        # Add advanced features
        if contract.advanced_features:
            advanced_features_dict = {}

            if contract.advanced_features.circuit_breaker:
                cb = contract.advanced_features.circuit_breaker
                advanced_features_dict["circuit_breaker"] = {
                    "enabled": cb.enabled,
                    "failure_threshold": cb.failure_threshold,
                    "recovery_timeout_ms": cb.recovery_timeout_ms,
                    "half_open_max_calls": cb.half_open_max_calls,
                }

            if contract.advanced_features.retry_policy:
                rp = contract.advanced_features.retry_policy
                advanced_features_dict["retry_policy"] = {
                    "enabled": rp.enabled,
                    "max_attempts": rp.max_attempts,
                    "initial_delay_ms": rp.initial_delay_ms,
                    "max_delay_ms": rp.max_delay_ms,
                    "backoff_multiplier": rp.backoff_multiplier,
                }

            if contract.advanced_features.observability:
                obs = contract.advanced_features.observability
                advanced_features_dict["observability"] = {
                    "tracing": obs.tracing,
                    "metrics": obs.metrics,
                    "logging": obs.logging,
                }

            if contract.advanced_features.dead_letter_queue:
                dlq = contract.advanced_features.dead_letter_queue
                advanced_features_dict["dead_letter_queue"] = {
                    "enabled": dlq.enabled,
                    "max_retries": dlq.max_retries,
                    "topic_suffix": dlq.topic_suffix,
                    "retry_delay_ms": dlq.retry_delay_ms,
                    "alert_threshold": dlq.alert_threshold,
                }

            if contract.advanced_features.transactions:
                tx = contract.advanced_features.transactions
                advanced_features_dict["transactions"] = {
                    "enabled": tx.enabled,
                    "isolation_level": tx.isolation_level,
                    "timeout_seconds": tx.timeout_seconds,
                    "rollback_on_error": tx.rollback_on_error,
                }

            if contract.advanced_features.security_validation:
                sv = contract.advanced_features.security_validation
                advanced_features_dict["security_validation"] = {
                    "enabled": sv.enabled,
                    "sanitize_inputs": sv.sanitize_inputs,
                    "sanitize_logs": sv.sanitize_logs,
                    "validate_sql": sv.validate_sql,
                }

            contract_dict["advanced_features"] = advanced_features_dict

        # Add subcontracts placeholder
        contract_dict["subcontracts"] = {
            contract.node_type: {
                "operations": [
                    "initialize",
                    "cleanup",
                    f"execute_{contract.node_type}",
                ]
            }
        }

        # Serialize to YAML
        yaml_str = yaml.dump(
            contract_dict,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=80,
        )

        # Add header comment
        header = f"""# ONEX v2.0 Contract - Auto-generated by ContractInferencer
# Generated from: {contract.name}
# DO NOT EDIT - Regenerate using ContractInferencer

"""

        return header + yaml_str

    async def cleanup(self) -> None:
        """Cleanup LLM node resources."""
        if self.llm_node:
            await self.llm_node.cleanup()


__all__ = ["ContractInferencer", "ModelNodeAnalysis", "ModelMixinConfigInference"]
