#!/usr/bin/env python3
"""
Template Engine for ONEX Code Generation.

Generates ONEX-compliant node code from templates using Jinja2.

Generates:
- node.py: Main node implementation
- contract.yaml: ONEX v2.0 contract
- models/*.py: Pydantic data models
- tests/*.py: Unit and integration tests
- __init__.py: Module initialization
- README.md: Node documentation

IMPORTANT: Does NOT generate main_standalone.py
- ONEX v2.0 nodes are invoked via orchestration, NOT as standalone REST APIs
- main_standalone.py is a LEGACY pattern for backward compatibility only
- New nodes should ONLY have node.py as the entry point
- If REST API access is needed, use a dedicated API gateway/orchestrator

ONEX v2.0 Compliance:
- Suffix-based naming conventions
- Contract-first architecture
- Event-driven patterns
- Comprehensive error handling
"""

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from jinja2 import Environment, FileSystemLoader
from omnibase_core.enums.enum_node_type import EnumNodeType as CoreEnumNodeType
from omnibase_core.models.contracts.model_contract_compute import ModelContractCompute
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.contracts.model_contract_orchestrator import (
    ModelContractOrchestrator,
)
from omnibase_core.models.contracts.model_contract_reducer import ModelContractReducer
from omnibase_core.primitives.model_semver import ModelSemVer
from pydantic import BaseModel, Field, ValidationError

from .contract_introspector import ContractIntrospector
from .failure_analyzer import FailureAnalyzer, ModelFailureAnalysis
from .node_classifier import EnumNodeType, ModelClassificationResult
from .prd_analyzer import ModelPRDRequirements
from .test_executor import ModelTestResults, TestExecutor

logger = logging.getLogger(__name__)


class ModelGeneratedArtifacts(BaseModel):
    """
    Generated code artifacts from template engine.

    Contains all generated files and metadata.

    NOTE: This model does NOT include main_standalone.py - ONEX v2.0 nodes
    are invoked via orchestration, not as standalone REST APIs. The
    main_standalone.py pattern is deprecated for new nodes.
    """

    # Generated files (path -> content)
    node_file: str = Field(..., description="Main node implementation (node.py)")
    contract_file: str = Field(..., description="ONEX contract (contract.yaml)")
    init_file: str = Field(..., description="Module init (__init__.py)")

    # Optional generated files
    models: dict[str, str] = Field(
        default_factory=dict, description="Data models (filename -> content)"
    )
    tests: dict[str, str] = Field(
        default_factory=dict, description="Test files (filename -> content)"
    )
    documentation: dict[str, str] = Field(
        default_factory=dict, description="Documentation files (filename -> content)"
    )

    # Metadata
    node_type: str = Field(..., description="Generated node type")
    node_name: str = Field(
        ..., description="Node class name (e.g., NodePostgresCRUDEffect)"
    )
    service_name: str = Field(..., description="Service name (e.g., postgres_crud)")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # File paths (for writing to disk)
    output_directory: Optional[Path] = None

    # Test execution results (if run_tests=True)
    test_results: Optional[ModelTestResults] = Field(
        None, description="Test execution results"
    )
    failure_analysis: Optional[ModelFailureAnalysis] = Field(
        None, description="Failure analysis (if tests failed)"
    )

    def get_all_files(self) -> dict[str, str]:
        """Get all generated files as {filename: content} dict."""
        files = {
            "node.py": self.node_file,
            "contract.yaml": self.contract_file,
            "__init__.py": self.init_file,
        }

        # Add models
        for filename, content in self.models.items():
            files[f"models/{filename}"] = content

        # Add tests
        for filename, content in self.tests.items():
            files[f"tests/{filename}"] = content

        # Add documentation
        for filename, content in self.documentation.items():
            files[filename] = content

        return files


class TemplateEngine:
    """
    Template-based code generation engine for ONEX nodes.

    Uses Jinja2 templates to generate ONEX-compliant code with:
    - Proper naming conventions (NodeXxxYyy suffix pattern)
    - Complete contract definitions
    - Comprehensive error handling
    - Event-driven patterns
    - Test scaffolding
    """

    def __init__(
        self,
        templates_directory: Optional[Path] = None,
        enable_inline_templates: bool = True,
    ):
        """
        Initialize template engine.

        Args:
            templates_directory: Path to templates directory
            enable_inline_templates: Enable inline template fallback
        """
        if templates_directory and templates_directory.exists():
            self.template_loader = FileSystemLoader(str(templates_directory))
            self.env = Environment(loader=self.template_loader, autoescape=False)

            # Register custom Jinja2 filters
            self._register_custom_filters()
        else:
            self.env = None

        self.enable_inline_templates = enable_inline_templates

        # Initialize contract introspector for validation
        self.introspector = ContractIntrospector()

    def _register_custom_filters(self) -> None:
        """Register custom Jinja2 filters for template rendering."""
        if not self.env:
            return

        # Filter: to_snake_case - Convert CamelCase to snake_case
        def to_snake_case(text: str) -> str:
            """Convert CamelCase to snake_case."""
            # Insert underscore before uppercase letters and convert to lowercase
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", text)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

        # Filter: sort_imports - Sort import statements
        def sort_imports(imports: list[str]) -> list[str]:
            """Sort import statements alphabetically."""
            return sorted(imports)

        # Filter: indent - Indent code block by n spaces
        def indent_filter(text: str, spaces: int = 4) -> str:
            """Indent each line of text by specified number of spaces."""
            indent_str = " " * spaces
            return "\n".join(
                indent_str + line if line.strip() else line for line in text.split("\n")
            )

        # Filter: repr - Python repr() for values
        def repr_filter(value: Any) -> str:
            """Convert value to Python repr() string."""
            return repr(value)

        # Register filters
        self.env.filters["to_snake_case"] = to_snake_case
        self.env.filters["sort_imports"] = sort_imports
        self.env.filters["indent"] = indent_filter
        self.env.filters["repr"] = repr_filter

        logger.debug(
            "Registered custom Jinja2 filters: to_snake_case, sort_imports, indent, repr"
        )

    def _validate_service_name(self, service_name: str) -> None:
        """
        Validate service name for security and correctness.

        Ensures service_name follows Python naming conventions and prevents
        code injection attacks in generated code.

        Args:
            service_name: Service name to validate

        Raises:
            ValueError: If service_name is invalid or potentially malicious

        Examples:
            Valid names: "postgres_crud", "api_handler", "data_processor"
            Invalid names: "", "Abc", "123abc", "abc-def", "abc def", "a"*100

        Security:
            - Prevents code injection via template variable substitution
            - Ensures generated Python code uses valid identifiers
            - Guards against filesystem traversal attacks
        """
        if not service_name:
            raise ValueError("service_name cannot be empty")

        # Validate format: lowercase letters, numbers, underscores only
        # Must start with a lowercase letter
        if not re.match(r"^[a-z][a-z0-9_]*$", service_name):
            raise ValueError(
                f"Invalid service_name '{service_name}': must start with lowercase letter "
                f"and contain only lowercase letters, numbers, and underscores"
            )

        # Prevent excessively long names that could cause issues
        if len(service_name) > 64:
            raise ValueError(
                f"service_name too long: {len(service_name)} chars (max 64)"
            )

        # Prevent single-character names (too generic)
        if len(service_name) < 2:
            raise ValueError(
                f"service_name too short: '{service_name}' (minimum 2 characters)"
            )

        logger.debug(f"Validated service_name: {service_name}")

    def _validate_domain(self, domain: str) -> None:
        """
        Validate domain name for security and correctness.

        Args:
            domain: Domain name to validate

        Raises:
            ValueError: If domain is invalid

        Examples:
            Valid domains: "api_development", "data_processing", "ml_training"
            Invalid domains: "", "API-Dev", "domain with spaces"
        """
        if not domain:
            raise ValueError("domain cannot be empty")

        if not re.match(r"^[a-z][a-z0-9_]*$", domain):
            raise ValueError(
                f"Invalid domain '{domain}': must start with lowercase letter "
                f"and contain only lowercase letters, numbers, and underscores"
            )

        if len(domain) > 64:
            raise ValueError(f"domain too long: {len(domain)} chars (max 64)")

        logger.debug(f"Validated domain: {domain}")

    def _validate_string_list(
        self, items: list[str], field_name: str, max_items: int = 100
    ) -> None:
        """
        Validate a list of string values for security.

        Args:
            items: List of strings to validate
            field_name: Name of the field being validated (for error messages)
            max_items: Maximum number of items allowed

        Raises:
            ValueError: If any item is invalid or list is too long
        """
        if not isinstance(items, list):
            raise ValueError(f"{field_name} must be a list")

        if len(items) > max_items:
            raise ValueError(
                f"{field_name} has too many items: {len(items)} (max {max_items})"
            )

        for idx, item in enumerate(items):
            if not isinstance(item, str):
                raise ValueError(
                    f"{field_name}[{idx}] must be a string, got {type(item).__name__}"
                )

            # Prevent excessively long strings
            if len(item) > 500:
                raise ValueError(
                    f"{field_name}[{idx}] too long: {len(item)} chars (max 500)"
                )

            # Prevent null bytes and other control characters
            if "\x00" in item or any(ord(c) < 32 and c not in "\n\t" for c in item):
                raise ValueError(
                    f"{field_name}[{idx}] contains invalid control characters"
                )

        logger.debug(f"Validated {field_name}: {len(items)} items")

    def _validate_contract_yaml(
        self,
        contract_yaml: str,
        node_name: str,
        node_type: EnumNodeType,
    ) -> None:
        """
        Validate generated contract YAML against omnibase_core schema.

        This validation catches missing required fields and structural issues
        before files are written, preventing broken contracts from reaching users.

        Args:
            contract_yaml: Generated contract YAML string
            node_name: Node name for error messages
            node_type: Node type for context in error messages

        Raises:
            ValueError: If contract is invalid with detailed error message

        Validation checks:
            1. YAML is well-formed (no syntax errors)
            2. Contains all required fields (name, version, description, etc.)
            3. Field types are correct (version is dict, name is string, etc.)
            4. Node type can be converted to CoreEnumNodeType

        Example error messages:
            - "Generated contract invalid: missing required fields: ['input_model', 'output_model']"
            - "Generated contract is not valid YAML: while parsing a block mapping..."
            - "Invalid node_type value in contract: 'effect' (must be uppercase EFFECT)"
        """
        try:
            # Step 1: Parse YAML
            contract_dict = yaml.safe_load(contract_yaml)

            if not isinstance(contract_dict, dict):
                raise ValueError(
                    f"Contract YAML must parse to a dictionary, got {type(contract_dict).__name__}"
                )

            # Step 2: Convert node_type string to enum for validation
            # The YAML has node_type as uppercase string (e.g., "EFFECT")
            # We need to convert it to CoreEnumNodeType enum
            node_type_str = contract_dict.get("node_type", "")

            if not node_type_str:
                raise ValueError("Contract missing 'node_type' field")

            try:
                # Convert string to enum (e.g., "EFFECT" -> CoreEnumNodeType.EFFECT)
                contract_dict["node_type"] = CoreEnumNodeType[node_type_str.upper()]
            except KeyError:
                raise ValueError(
                    f"Invalid node_type value in contract: '{node_type_str}' "
                    f"(must be one of: EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)"
                )

            # Step 2.5: Convert version dict to ModelSemVer object
            # The YAML has version as dict (e.g., {"major": 1, "minor": 0, "patch": 0})
            # We need to convert it to ModelSemVer object
            version_data = contract_dict.get("version")
            if version_data and isinstance(version_data, dict):
                try:
                    contract_dict["version"] = ModelSemVer(
                        major=version_data.get("major", 1),
                        minor=version_data.get("minor", 0),
                        patch=version_data.get("patch", 0),
                    )
                except Exception as e:
                    raise ValueError(
                        f"Invalid version format in contract: {version_data}. "
                        f"Expected dict with major/minor/patch keys. Error: {e}"
                    )
            elif not version_data:
                raise ValueError("Contract missing 'version' field")

            # Also convert node_version if present (legacy compatibility)
            node_version_data = contract_dict.get("node_version")
            if node_version_data and isinstance(node_version_data, dict):
                try:
                    contract_dict["node_version"] = ModelSemVer(
                        major=node_version_data.get("major", 1),
                        minor=node_version_data.get("minor", 0),
                        patch=node_version_data.get("patch", 0),
                    )
                except Exception:
                    # node_version is optional, so we can skip conversion errors
                    pass

            # Step 3: Validate against appropriate contract model based on node type
            # Select the correct contract class for validation
            contract_class_map = {
                CoreEnumNodeType.EFFECT: ModelContractEffect,
                CoreEnumNodeType.COMPUTE: ModelContractCompute,
                CoreEnumNodeType.REDUCER: ModelContractReducer,
                CoreEnumNodeType.ORCHESTRATOR: ModelContractOrchestrator,
            }

            contract_class = contract_class_map.get(contract_dict["node_type"])
            if not contract_class:
                raise ValueError(
                    f"Unknown node_type enum: {contract_dict['node_type']}"
                )

            try:
                # Validate using the appropriate contract model
                contract_class.model_validate(contract_dict)
            except ValidationError as e:
                # Extract missing fields for helpful error message
                missing_fields = [
                    ".".join(str(loc) for loc in err["loc"])
                    for err in e.errors()
                    if err["type"] == "missing"
                ]

                type_errors = [
                    {
                        "field": ".".join(str(loc) for loc in err["loc"]),
                        "expected": err.get("msg", ""),
                        "type": err["type"],
                    }
                    for err in e.errors()
                    if err["type"] != "missing"
                ]

                # Build detailed error message
                error_parts = [f"Generated contract validation failed for {node_name}:"]

                if missing_fields:
                    error_parts.append(
                        f"\n  Missing required fields: {', '.join(missing_fields)}"
                    )

                if type_errors:
                    error_parts.append("\n  Type/validation errors:")
                    for err in type_errors[
                        :5
                    ]:  # Show max 5 to avoid overwhelming output
                        error_parts.append(
                            f"    - Field '{err['field']}': {err['expected']}"
                        )

                    if len(type_errors) > 5:
                        error_parts.append(f"    - ... and {len(type_errors) - 5} more")

                raise ValueError("\n".join(error_parts)) from e

            logger.info(
                f"Contract validation passed for {node_name} ({node_type.value})"
            )

        except yaml.YAMLError as e:
            # YAML parsing failed
            error_lines = str(e).split("\n")[:3]  # First 3 lines of YAML error
            raise ValueError(
                f"Generated contract for {node_name} is not valid YAML:\n"
                + "\n".join(f"  {line}" for line in error_lines)
                + f"\n\nFirst 500 chars of contract:\n{contract_yaml[:500]}"
            ) from e

    async def generate(
        self,
        requirements: ModelPRDRequirements,
        classification: ModelClassificationResult,
        output_directory: Path,
        run_tests: bool = False,
        strict_mode: bool = False,
    ) -> ModelGeneratedArtifacts:
        """
        Generate complete node implementation from templates.

        Args:
            requirements: Extracted PRD requirements
            classification: Node type classification
            output_directory: Target directory for generated files
            run_tests: Execute generated tests after code generation
            strict_mode: Raise exception if tests fail (default: attach to artifacts)

        Returns:
            ModelGeneratedArtifacts with all generated code and test results

        Raises:
            ValueError: If strict_mode=True and tests fail

        Example:
            >>> engine = TemplateEngine()
            >>> artifacts = await engine.generate(
            ...     requirements=reqs,
            ...     classification=classification,
            ...     output_directory=Path("./generated_nodes/postgres_crud"),
            ...     run_tests=True,
            ...     strict_mode=False,
            ... )
            >>> assert artifacts.node_name == "NodePostgresCRUDEffect"
            >>> assert "async def execute_effect" in artifacts.node_file
            >>> if artifacts.test_results:
            ...     print(f"Tests: {artifacts.test_results.passed}/{artifacts.test_results.total} passed")
        """
        # Build template context
        context = self._build_template_context(requirements, classification)

        # Validate template context has all required data for contract generation
        is_valid, missing_data = self.validate_template_context(
            context, classification.node_type
        )
        if not is_valid:
            # Log warning but don't fail - allow generation to proceed
            # The contract validation will catch actual issues
            logger.warning(
                f"Template context may be missing data for {classification.node_type.value} contract: {missing_data}. "
                f"Contract generation will proceed but may fail validation."
            )

        # Generate node file
        node_content = self._generate_node_file(
            classification.node_type, classification.template_name, context
        )

        # Generate contract file
        contract_content = self._generate_contract_file(
            classification.node_type, context
        )

        # Validate generated contract before proceeding
        # This catches missing fields and structural issues early
        try:
            self._validate_contract_yaml(
                contract_yaml=contract_content,
                node_name=context["node_class_name"],
                node_type=classification.node_type,
            )
        except ValueError as e:
            # Re-raise with additional context about code generation
            raise ValueError(
                f"Contract generation failed validation:\n{e!s}\n\n"
                f"This is a code generation bug - the template is missing required fields. "
                f"Generated contract needs to be updated to include all omnibase_core requirements."
            ) from e

        # Generate init file
        init_content = self._generate_init_file(context)

        # Generate models
        models = self._generate_models(requirements, context)

        # Generate tests
        tests = self._generate_tests(classification.node_type, context)

        # Generate documentation
        documentation = self._generate_documentation(requirements, context)

        # Create artifacts (before test execution)
        artifacts = ModelGeneratedArtifacts(
            node_file=node_content,
            contract_file=contract_content,
            init_file=init_content,
            models=models,
            tests=tests,
            documentation=documentation,
            node_type=classification.node_type.value,
            node_name=context["node_class_name"],
            service_name=context["service_name"],
            output_directory=output_directory,
        )

        # Execute tests if requested
        if run_tests:
            logger.info(f"Executing generated tests for {artifacts.node_name}")
            try:
                # Write files to disk first (required for pytest)
                self._write_artifacts_to_disk(artifacts, output_directory)

                # Run tests
                test_executor = TestExecutor()
                test_results = await test_executor.run_tests(
                    output_directory=output_directory,
                    test_types=["unit", "integration"],
                )

                artifacts.test_results = test_results

                # Analyze failures if any
                if not test_results.is_passing:
                    logger.warning(
                        f"Generated code has {test_results.failed} failing tests"
                    )
                    failure_analyzer = FailureAnalyzer()
                    failure_analysis = failure_analyzer.analyze(test_results)
                    artifacts.failure_analysis = failure_analysis

                    # Log failure report
                    logger.error(f"Test failures detected:\n{failure_analysis.summary}")
                    for fix in failure_analysis.recommended_fixes[:3]:
                        logger.error(f"  Recommended fix: {fix}")

                    # Strict mode: raise exception
                    if strict_mode:
                        raise ValueError(
                            f"Generated code failed tests ({test_results.failed}/{test_results.total} failures).\n"
                            f"{failure_analysis.summary}\n\n"
                            f"Recommended fixes:\n"
                            + "\n".join(
                                f"  - {fix}"
                                for fix in failure_analysis.recommended_fixes
                            )
                        )
                else:
                    logger.info(
                        f"All tests passed! ({test_results.passed}/{test_results.total})"
                    )

            except Exception as e:
                logger.error(f"Test execution failed: {e}")
                # In non-strict mode, attach error to artifacts
                if not strict_mode:
                    artifacts.test_results = ModelTestResults(
                        passed=0,
                        failed=0,
                        skipped=0,
                        total=0,
                        duration_seconds=0.0,
                        errors=[str(e)],
                        exit_code=-1,
                        test_types_run=[],
                    )
                else:
                    raise

        return artifacts

    def validate_template_context(
        self, context: dict[str, Any], node_type: EnumNodeType
    ) -> tuple[bool, list[str]]:
        """
        Validate template context has all required data for contract generation.

        Uses ContractIntrospector to verify the template context contains
        sufficient data to generate a valid contract YAML for the node type.

        This validation prevents generating contracts with missing required fields.

        Args:
            context: Template context dict (from _build_template_context)
            node_type: Node type being generated (EFFECT, COMPUTE, etc.)

        Returns:
            Tuple of (is_valid: bool, missing_data: list[str])
            - is_valid: True if context has all necessary data
            - missing_data: List of missing required data keys (empty if valid)

        Example:
            >>> engine = TemplateEngine()
            >>> context = {'service_name': 'test', ...}
            >>> is_valid, missing = engine.validate_template_context(
            ...     context, EnumNodeType.EFFECT
            ... )
            >>> if not is_valid:
            ...     print(f"Missing data: {missing}")
        """
        return self.introspector.validate_template_context_for_node_type(
            node_type, context
        )

    def _select_base_class(
        self,
        requirements: ModelPRDRequirements,
        classification: ModelClassificationResult,
    ) -> dict[str, Any]:
        """
        Select between convenience wrapper and custom mixin composition.

        Selection Logic:
        - **Convenience Wrapper (Simple)**: Standard nodes with typical requirements
          - Uses ModelService* wrappers (ModelServiceEffect, ModelServiceCompute, etc.)
          - Includes: MixinNodeService, base class, standard mixins (health, metrics, events/caching)

        - **Custom Composition (Complex)**: Specialized nodes with unique requirements
          - Uses base classes (NodeEffect, NodeCompute, etc.)
          - Manually composes specific mixins needed

        Criteria for Custom Composition:
        1. Complexity > 10 (complex business logic)
        2. Custom mixins specified (retry, circuit breaker, validation, security)
        3. Explicit "no service mode" requirement
        4. High-performance requirements (no overhead tolerance)

        Args:
            requirements: PRD requirements
            classification: Node classification result

        Returns:
            Dictionary with:
            - use_convenience_wrapper (bool): True for ModelService*, False for custom
            - base_class_name (str): Class to inherit from
            - mixin_list (list[str]): List of mixin names (empty for convenience wrapper)
            - import_paths (dict): Import statements by category
        """
        # Calculate complexity
        operation_count = len(requirements.operations)
        feature_count = len(requirements.features)
        dependency_count = len(requirements.dependencies)

        # Simple complexity calculation
        total_complexity = operation_count + feature_count + dependency_count

        # Check for custom mixin keywords in requirements
        description_lower = requirements.business_description.lower()
        needs_retry = "retry" in description_lower or "fault" in description_lower
        needs_circuit_breaker = (
            "circuit" in description_lower or "resilient" in description_lower
        )
        needs_validation = (
            "validate" in description_lower or "validation" in description_lower
        )
        needs_security = (
            "secure" in description_lower or "security" in description_lower
        )

        # Check for explicit "no service mode" requirement
        no_service_mode = (
            "one-shot" in description_lower
            or "ephemeral" in description_lower
            or "temporary" in description_lower
        )

        # Decision: Use convenience wrapper by default, custom composition for edge cases
        use_convenience_wrapper = True
        custom_mixins = []

        # Force custom composition if:
        if (
            total_complexity > 10  # Complex logic
            or needs_retry  # Needs retry mixin
            or needs_circuit_breaker  # Needs circuit breaker
            or needs_validation  # Needs validation mixin
            or needs_security  # Needs security mixin
            or no_service_mode  # No service mode needed
        ):
            use_convenience_wrapper = False

            # Build custom mixin list
            if needs_retry:
                custom_mixins.append("MixinRetry")
            if needs_circuit_breaker:
                custom_mixins.append("MixinCircuitBreaker")
            if needs_validation:
                custom_mixins.append("MixinValidation")
            if needs_security:
                custom_mixins.append("MixinSecurity")

            # Add standard mixins
            custom_mixins.append("MixinHealthCheck")
            custom_mixins.append("MixinMetrics")

        # Determine base class name
        node_type_str = classification.node_type.value.capitalize()

        if use_convenience_wrapper:
            base_class_name = f"ModelService{node_type_str}"
            mixin_list = []  # Convenience wrappers include mixins automatically
        else:
            base_class_name = f"Node{node_type_str}"
            mixin_list = custom_mixins

        # Build import paths
        import_paths = {}

        if use_convenience_wrapper:
            # Single import for convenience wrapper
            # Note: Using local implementations until omnibase_core v0.2.0 is released
            import_paths["convenience_wrapper"] = [
                f"from omninode_bridge.utils.node_services import {base_class_name}"
            ]
        else:
            # Base class import
            import_paths["base_class"] = [
                f"from omnibase_core.nodes.node_{classification.node_type.value} import {base_class_name}"
            ]

            # Mixin imports
            import_paths["mixins"] = []
            for mixin_name in mixin_list:
                mixin_module = mixin_name.replace("Mixin", "mixin_").lower()
                import_paths["mixins"].append(
                    f"from omnibase_core.mixins.{mixin_module} import {mixin_name}"
                )

        logger.info(
            f"Mixin selection: use_convenience_wrapper={use_convenience_wrapper}, "
            f"base_class={base_class_name}, mixins={mixin_list}"
        )

        return {
            "use_convenience_wrapper": use_convenience_wrapper,
            "base_class_name": base_class_name,
            "mixin_list": mixin_list,
            "import_paths": import_paths,
            "selection_reasoning": {
                "complexity": total_complexity,
                "needs_retry": needs_retry,
                "needs_circuit_breaker": needs_circuit_breaker,
                "needs_validation": needs_validation,
                "needs_security": needs_security,
                "no_service_mode": no_service_mode,
            },
        }

    def _build_template_context(
        self,
        requirements: ModelPRDRequirements,
        classification: ModelClassificationResult,
    ) -> dict[str, Any]:
        """
        Build Jinja2 template context from requirements and classification.

        Validates all user inputs for security before building context.

        Raises:
            ValueError: If any input validation fails
        """
        # Validate inputs for security before using them
        service_name = requirements.service_name
        self._validate_service_name(service_name)

        # Validate domain
        self._validate_domain(requirements.domain)

        # Validate string lists
        self._validate_string_list(requirements.operations, "operations", max_items=50)
        self._validate_string_list(requirements.features, "features", max_items=50)

        # Select base class and mixins (convenience wrapper vs custom composition)
        mixin_selection = self._select_base_class(requirements, classification)

        # Get node type
        node_type = classification.node_type.value

        # Convert service_name to PascalCase
        # Example: postgres_crud -> PostgresCrud
        pascal_name = "".join(word.capitalize() for word in service_name.split("_"))

        # Node class name: Node<PascalName><Type>
        # Example: NodePostgresCrudEffect
        node_class_name = (
            f"Node{pascal_name}{classification.node_type.value.capitalize()}"
        )

        # Build version dict for new contract format
        version_dict = {"major": 1, "minor": 0, "patch": 0}

        # Build package path (e.g., "omninode_bridge.nodes.service_name.v1_0_0.node")
        package_path = f"omninode_bridge.nodes.{service_name}.v1_0_0.node"

        # Input/output model names
        input_model = f"Model{pascal_name}Request"
        output_model = f"Model{pascal_name}Response"

        # Build imports structure
        imports = {
            "standard_library": [
                "import logging",
                "from typing import Any, Dict, List, Optional",
                "from datetime import UTC, datetime",
                "import uuid",
            ],
            "third_party": [],
            "omnibase_core": [
                "from omnibase_core.models.core import ModelContainer",
                f"from omnibase_core.nodes.node_{node_type} import Node{node_type.capitalize()}",
                "from omnibase_core import ModelOnexError, EnumCoreErrorCode",
                "from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel",
                "from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event",
            ],
            "omnibase_mixins": [],
            "project_local": [],
        }

        # Default mixin-related data (will be populated when mixin support is added)
        enabled_mixins = []
        mixin_configs = {}
        mixin_descriptions = {}
        base_classes = [f"Node{node_type.capitalize()}"]

        # Health check components (example - will be extracted from contract)
        health_check_components = []

        # IO operations (for Effect nodes)
        io_operations = []
        if classification.node_type == EnumNodeType.EFFECT:
            for operation in requirements.operations[:3]:  # Limit to 3 operations
                io_operations.append(
                    {
                        "name": operation.lower().replace(" ", "_"),
                        "description": f"Execute {operation} operation",
                        "input_model": input_model,
                        "output_model": output_model,
                        "operation_type": "database_query",
                        "atomic": True,
                        "timeout_seconds": 30,
                        "validation_enabled": True,
                    }
                )

        # Compute operations (for Compute nodes)
        compute_operations = []
        if classification.node_type == EnumNodeType.COMPUTE:
            for operation in requirements.operations[:3]:
                compute_operations.append(
                    {
                        "name": operation.lower().replace(" ", "_"),
                        "description": f"Compute {operation}",
                        "input_model": input_model,
                        "output_model": output_model,
                    }
                )

        # Reduction operations (for Reducer nodes)
        reduction_operations = []
        aggregation_types = []
        if classification.node_type == EnumNodeType.REDUCER:
            for operation in requirements.operations[:3]:
                reduction_operations.append(
                    {
                        "name": operation.lower().replace(" ", "_"),
                        "description": f"Aggregate {operation} data",
                        "input_model": input_model,
                        "output_model": output_model,
                    }
                )
            aggregation_types = [
                {"name": "namespace_grouping", "description": "Group by namespace"},
                {"name": "time_window", "description": "Time-based windowing"},
            ]

        # Workflows (for Orchestrator nodes)
        workflows = []
        if classification.node_type == EnumNodeType.ORCHESTRATOR:
            for operation in requirements.operations[:3]:
                workflows.append(
                    {
                        "name": operation.lower().replace(" ", "_"),
                        "description": f"Orchestrate {operation} workflow",
                        "input_model": input_model,
                        "output_model": output_model,
                    }
                )

        # Advanced features (example - will be extracted from contract)
        advanced_features = {}

        # Testing configuration
        testing = {
            "unit_test_coverage": 85,
            "integration_tests_required": True,
        }

        return {
            # Names
            "service_name": service_name,
            "node_name": pascal_name + node_type.capitalize(),
            "class_name": node_class_name,
            "node_class_name": node_class_name,
            "node_type": node_type,
            "node_type_upper": node_type.upper(),  # EFFECT, ORCHESTRATOR, etc.
            "pascal_name": pascal_name,
            # Requirements
            "description": requirements.business_description,
            "business_description": requirements.business_description,
            "operations": requirements.operations,
            "features": requirements.features,
            "domain": requirements.domain,
            # Dependencies
            "dependencies": requirements.dependencies,
            "performance_requirements": requirements.performance_requirements,
            # Intelligence
            "best_practices": requirements.best_practices,
            "code_examples": requirements.code_examples,
            # Classification
            "template_name": classification.template_name,
            "template_variant": classification.template_variant,
            "confidence": classification.confidence,
            # Metadata
            "generated_at": datetime.now(UTC).isoformat(),
            "generation_timestamp": datetime.now(UTC).isoformat(),
            "version": "1.0.0",
            "version_dict": version_dict,  # For new contract format
            # Package and models (for new contract format)
            "package_path": package_path,
            "input_model": input_model,
            "output_model": output_model,
            # Models
            "data_models": requirements.data_models,
            # Mixin Selection (NEW - for convenience wrapper vs custom composition)
            "use_convenience_wrapper": mixin_selection["use_convenience_wrapper"],
            "base_class_name": mixin_selection["base_class_name"],
            "mixin_list": mixin_selection["mixin_list"],
            "mixin_import_paths": mixin_selection["import_paths"],
            "mixin_selection_reasoning": mixin_selection["selection_reasoning"],
            # Template-specific data
            "imports": imports,
            "base_classes": base_classes,
            "enabled_mixins": enabled_mixins,
            "mixin_configs": mixin_configs,
            "mixin_descriptions": mixin_descriptions,
            "health_check_components": health_check_components,
            "io_operations": io_operations,
            "compute_operations": compute_operations,
            "reduction_operations": reduction_operations,
            "aggregation_types": aggregation_types,
            "workflows": workflows,
            "advanced_features": advanced_features,
            "testing": testing,
        }

    def _generate_node_file(
        self, node_type: EnumNodeType, template_name: str, context: dict[str, Any]
    ) -> str:
        """Generate main node implementation file."""
        # Try to load template from filesystem
        if self.env:
            try:
                # Load node template by type (e.g., node_templates/node_effect.py.j2)
                template = self.env.get_template(
                    f"node_templates/node_{node_type.value}.py.j2"
                )
                return template.render(**context)
            except Exception as e:
                logger.warning(
                    f"Failed to load template from filesystem for {node_type.value}: {e}. "
                    f"Falling back to inline template."
                )

        # Fallback: Use inline template
        if self.enable_inline_templates:
            return self._get_inline_node_template(node_type, context)

        raise ValueError(
            f"No template found for {node_type.value} and inline templates disabled"
        )

    def _generate_contract_file(
        self, node_type: EnumNodeType, context: dict[str, Any]
    ) -> str:
        """Generate ONEX v2.0 contract YAML."""
        if self.env:
            try:
                # Load unified contract template for all node types
                template = self.env.get_template("node_templates/contract.yaml.j2")
                return template.render(**context)
            except Exception as e:
                logger.warning(
                    f"Failed to load contract template from filesystem for {node_type.value}: {e}. "
                    f"Falling back to inline template."
                )

        # Inline contract template
        return self._get_inline_contract_template(node_type, context)

    def _generate_init_file(self, context: dict[str, Any]) -> str:
        """Generate __init__.py module file."""
        # Try to load template from filesystem
        if self.env:
            try:
                template = self.env.get_template("node_templates/__init__.py.j2")
                return template.render(**context)
            except Exception as e:
                logger.warning(
                    f"Failed to load __init__ template: {e}. Using inline template."
                )

        # Fallback: inline template
        return f'''"""
{context['service_name']} - {context['business_description']}

Generated: {context['generated_at']}
ONEX v2.0 Compliant
"""

from .node import {context['node_class_name']}

__all__ = ["{context['node_class_name']}"]
'''

    def _generate_models(
        self, requirements: ModelPRDRequirements, context: dict[str, Any]
    ) -> dict[str, str]:
        """Generate Pydantic data models."""
        models = {}

        # Generate models based on operations
        if "create" in requirements.operations or "update" in requirements.operations:
            models["model_request.py"] = self._get_request_model_template(context)

        if "read" in requirements.operations:
            models["model_response.py"] = self._get_response_model_template(context)

        # Add __init__.py for models directory
        model_names = [
            f"Model{context['pascal_name']}Request",
            f"Model{context['pascal_name']}Response",
        ]
        models[
            "__init__.py"
        ] = f'''"""Data models for {context['service_name']}."""

from .model_request import {model_names[0]}
from .model_response import {model_names[1]}

__all__ = {model_names}
'''

        return models

    def _build_test_contract_context(
        self, node_type: EnumNodeType, context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Build test contract context for Jinja2 test templates.

        The test templates expect a test_contract object with configuration
        for test generation including mocking, fixtures, and assertions.

        Args:
            node_type: Node type (EFFECT, COMPUTE, etc.)
            context: Main template context

        Returns:
            Test contract context dict
        """
        # Determine mock requirements based on node type
        mock_requirements = {
            "mock_database": node_type == EnumNodeType.EFFECT,
            "mock_kafka_producer": node_type
            in (EnumNodeType.EFFECT, EnumNodeType.REDUCER),
            "mock_http_clients": node_type == EnumNodeType.EFFECT,
            "mock_datetime": False,
            "mock_dependencies": [],
            "mock_external_services": [],
            "mock_return_values": {},
        }

        # Test configuration
        test_configuration = {
            "pytest_markers": ["slow", "onex"],
            "verbose_output": True,
            "environment_variables": {"ENVIRONMENT": "test"},
            # Note: test_data_directory is kept for documentation but templates now use
            # Path(__file__).parent / "data" for directory-independent path resolution
            "test_data_directory": "./tests/data",
            "use_test_database": node_type == EnumNodeType.EFFECT,
            "test_database_config": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db",
            },
            "timeout_seconds": 60,
            "parallel_execution": True,
            "parallel_workers": 4,
        }

        # Test targets - methods to test
        test_targets = [
            {
                "target_name": f"execute_{node_type.value}",
                "expected_behaviors": [
                    f"{node_type.value.capitalize()} executes successfully",
                    "Result is not None",
                    "Result contains expected fields",
                ],
                "assertions": [
                    'assert result.get("status") is not None',
                ],
                "input_parameters": {},
                "expected_outputs": {},
                "error_conditions": ["invalid input", "execution error"],
                "edge_cases": ["empty input", "large input"],
                "required_fixtures": [],
                "skip_test": False,
                "skip_reason": None,
            }
        ]

        return {
            "coverage_target": 85,
            "mock_requirements": mock_requirements,
            "test_configuration": test_configuration,
            "test_targets": test_targets,
            "use_async_tests": True,
            "include_docstrings": True,
            "parametrize_tests": False,
            "assertion_types": ["equality", "type", "exception"],
            "custom_assertions": [],
            "enforce_test_isolation": True,
            "enforce_deterministic_tests": True,
        }

    def _generate_tests(
        self, node_type: EnumNodeType, context: dict[str, Any]
    ) -> dict[str, str]:
        """Generate test files using Jinja2 templates or inline fallback."""
        tests = {}

        # Build test contract context for Jinja2 templates
        test_contract_context = self._build_test_contract_context(node_type, context)

        # Build module_path for imports in tests
        # e.g., "omninode_bridge.nodes.postgres_crud.v1_0_0.node"
        module_path = f"omninode_bridge.nodes.{context['service_name']}.v1_0_0.node"

        # Merge test contract into main context
        test_context = {
            **context,
            "test_contract": test_contract_context,
            "module_path": module_path,
            "node_name": context["node_class_name"],
            "fixtures": [],  # Custom fixtures can be added later
        }

        # Try to load Jinja2 templates
        if self.env:
            try:
                # Generate conftest.py
                conftest_template = self.env.get_template(
                    "test_templates/conftest.py.j2"
                )
                tests["conftest.py"] = conftest_template.render(**test_context)

                # Generate unit test
                unit_test_template = self.env.get_template(
                    "test_templates/test_unit.py.j2"
                )
                tests["test_node.py"] = unit_test_template.render(**test_context)

                # Generate integration test (for Effect nodes)
                if node_type == EnumNodeType.EFFECT:
                    integration_test_template = self.env.get_template(
                        "test_templates/test_integration.py.j2"
                    )
                    tests["test_integration.py"] = integration_test_template.render(
                        **test_context
                    )

                # Test __init__.py
                tests["__init__.py"] = '"""Tests for generated node."""\n'

                logger.info(f"Generated {len(tests)} test files using Jinja2 templates")
                return tests

            except Exception as e:
                logger.warning(
                    f"Failed to load Jinja2 test templates: {e}. "
                    f"Falling back to inline templates."
                )

        # Fallback: Use inline templates
        logger.info("Using inline test templates (Jinja2 not available)")

        # Unit test
        tests["test_node.py"] = self._get_unit_test_template(node_type, context)

        # Integration test (for Effect nodes)
        if node_type == EnumNodeType.EFFECT:
            tests["test_integration.py"] = self._get_integration_test_template(context)

        # Test __init__.py
        tests["__init__.py"] = '"""Tests for generated node."""\n'

        return tests

    def _generate_documentation(
        self, requirements: ModelPRDRequirements, context: dict[str, Any]
    ) -> dict[str, str]:
        """Generate documentation files."""
        readme_content = f"""# {context['node_class_name']}

## Overview

{requirements.business_description}

**Node Type**: {context['node_type']}
**Domain**: {requirements.domain}
**Generated**: {context['generated_at']}

## Operations

{chr(10).join(f"- `{op}`" for op in requirements.operations)}

## Features

{chr(10).join(f"- {feature}" for feature in requirements.features)}

## Performance Requirements

{chr(10).join(f"- {k}: {v}" for k, v in requirements.performance_requirements.items())}

## Usage

```python
from omninode_bridge.nodes.{context['service_name']}.v1_0_0 import {context['node_class_name']}
from omnibase_core.models.core import ModelContainer
from omnibase_core.models.contracts.model_contract_{context['node_type']} import ModelContract{context['node_type'].capitalize()}
from omnibase_core.enums.enum_node_type import EnumNodeType
from uuid import uuid4

# Initialize container with configuration
container = ModelContainer(
    value={{"environment": "production", "log_level": "INFO"}},
    container_type="config"
)
node = {context['node_class_name']}(container)

# Create contract with complete configuration
contract = ModelContract{context['node_type'].capitalize()}(
    name="{requirements.operations[0] if requirements.operations else 'example_operation'}",
    version={{"major": 1, "minor": 0, "patch": 0}},
    description="Execute {requirements.operations[0] if requirements.operations else 'operation'}",
    node_type=EnumNodeType.{context['node_type'].upper()},
    input_model="Model{context['node_class_name']}Input",
    output_model="Model{context['node_class_name']}Output",
    correlation_id=uuid4(),
    execution_id=uuid4()
)

# Execute the operation
result = await node.execute_{context['node_type']}(contract)
print(f"Result: {{result}}")
```

## Testing

```bash
pytest tests/test_node.py -v
```

## ONEX v2.0 Compliance

✅ Suffix-based naming: `{context['node_class_name']}`
✅ Contract-driven architecture
✅ Event-driven patterns
✅ Comprehensive error handling
✅ Performance monitoring

---

Generated by OmniNode Code Generation Pipeline
"""

        return {"README.md": readme_content}

    # ====================================
    # Inline Templates (Fallback)
    # ====================================

    def _generate_dependency_imports(self, dependencies: dict[str, str]) -> str:
        """
        Generate import statements for external dependencies.

        Args:
            dependencies: Dict of {package_name: version} from requirements

        Returns:
            String with import statements (one per line)

        Examples:
            >>> dependencies = {"hvac": "^1.0.0", "requests": "^2.28.0"}
            >>> imports = self._generate_dependency_imports(dependencies)
            >>> assert "import hvac" in imports
            >>> assert "import requests" in imports
        """
        if not dependencies:
            return "# No external dependencies specified"

        import_lines = ["# External dependencies"]

        # Common import patterns for known libraries
        import_patterns = {
            "hvac": "import hvac",
            "requests": "import requests",
            "aiohttp": "import aiohttp",
            "asyncpg": "import asyncpg",
            "psycopg2": "import psycopg2",
            "sqlalchemy": "from sqlalchemy import create_engine",
            "redis": "import redis",
            "boto3": "import boto3",
            "kafka": "from kafka import KafkaProducer, KafkaConsumer",
            "pydantic": "from pydantic import BaseModel, Field",
            "fastapi": "from fastapi import FastAPI, HTTPException",
        }

        for package_name in sorted(dependencies.keys()):
            # Use predefined pattern or default to simple import
            import_stmt = import_patterns.get(package_name, f"import {package_name}")
            import_lines.append(import_stmt)

        return "\n".join(import_lines)

    def _get_inline_node_template(
        self, node_type: EnumNodeType, context: dict[str, Any]
    ) -> str:
        """Get inline node template for given type."""
        if node_type == EnumNodeType.EFFECT:
            return self._get_effect_template(context)
        elif node_type == EnumNodeType.COMPUTE:
            return self._get_compute_template(context)
        elif node_type == EnumNodeType.REDUCER:
            return self._get_reducer_template(context)
        elif node_type == EnumNodeType.ORCHESTRATOR:
            return self._get_orchestrator_template(context)

        raise ValueError(f"Unknown node type: {node_type}")

    def _get_effect_template(self, context: dict[str, Any]) -> str:
        """Inline template for Effect nodes with introspection and registration."""
        # Generate dependency imports
        dependency_imports = self._generate_dependency_imports(
            context.get("dependencies", {})
        )

        # Generate base class imports based on mixin selection
        use_convenience_wrapper = context.get("use_convenience_wrapper", True)
        base_class_name = context.get("base_class_name", "ModelServiceEffect")
        mixin_list = context.get("mixin_list", [])
        mixin_import_paths = context.get("mixin_import_paths", {})

        # Build import statements
        if use_convenience_wrapper:
            # Convenience wrapper import (single line)
            base_imports = "\n".join(mixin_import_paths.get("convenience_wrapper", []))
            inheritance_line = f"class {context['node_class_name']}({base_class_name}):"
            mixin_description = f"Uses {base_class_name} convenience wrapper (includes MixinNodeService, NodeEffect, MixinHealthCheck, MixinEventBus, MixinMetrics)"
        else:
            # Custom composition imports
            base_imports = "\n".join(mixin_import_paths.get("base_class", []))
            if mixin_import_paths.get("mixins"):
                base_imports += "\n" + "\n".join(mixin_import_paths["mixins"])

            # Build inheritance chain
            inheritance_chain = [base_class_name] + mixin_list
            inheritance_line = (
                f"class {context['node_class_name']}({', '.join(inheritance_chain)}):"
            )
            mixin_description = f"Custom composition with {', '.join(mixin_list)}"

        return f'''#!/usr/bin/env python3
"""
{context['node_class_name']} - {context['business_description']}

ONEX v2.0 Effect Node with Registration & Introspection
Domain: {context['domain']}
Generated: {context['generated_at']}

Architecture: {mixin_description}
"""

import os
from typing import Any, Dict, List, Optional

from omnibase_core import ModelOnexError, EnumCoreErrorCode
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event

# Base class and mixin imports
{base_imports}

{dependency_imports}

MIXINS_AVAILABLE = True


{inheritance_line}
    """
    {context['business_description']}

    Operations:
{chr(10).join(f"    - {op}" for op in context['operations'])}

    Features:
{chr(10).join(f"    - {feature}" for feature in context['features'])}

    Capabilities:
    - Automatic node registration via introspection events
    - Health check endpoints
    - Consul service discovery integration
    """

    def __init__(self, container: ModelContainer) -> None:
        """Initialize {context['node_class_name']} with registration and introspection."""
        super().__init__(container)

        # Configuration
        # Access config from container.value (ModelContainer stores config in value field)
        self.config = container.value if isinstance(container.value, dict) else {{}}

        # Initialize health checks (if mixins available)
        if MIXINS_AVAILABLE:
            self.initialize_health_checks()
            self._register_component_checks()

            # Initialize introspection system
            self.initialize_introspection()

        emit_log_event(
            LogLevel.INFO,
            "{context['node_class_name']} initialized with registration support",
            {{
                "node_id": str(self.node_id),
                "mixins_available": MIXINS_AVAILABLE,
                "operations": {context['operations']},
                "features": {context['features']},
            }}
        )

    def _register_component_checks(self) -> None:
        """
        Register component health checks for this node.

        Override this method to add custom health checks specific to this node's dependencies.
        """
        # Base node runtime check is registered by HealthCheckMixin
        # Add custom checks here as needed
        pass

    async def execute_effect(self, contract: ModelContractEffect) -> Any:
        """
        Execute effect operation.

        Args:
            contract: Effect contract with operation parameters

        Returns:
            Operation result

        Raises:
            ModelOnexError: If operation fails
        """
        # IMPLEMENTATION REQUIRED
        pass

    async def _old_exception_handling_example(self) -> None:
        """Example exception handling (remove this method)."""
        try:
            pass
        except (ConnectionError, TimeoutError) as e:
            # Network/connection failures
            emit_log_event(
                LogLevel.ERROR,
                f"Network error during effect execution: {{e!s}}",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                    "error_type": type(e).__name__,
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.NETWORK_ERROR,
                message=f"Network error: {{e!s}}",
                details={{"original_error": str(e), "error_type": type(e).__name__}},
            ) from e

        except ValueError as e:
            # Invalid input/configuration
            emit_log_event(
                LogLevel.ERROR,
                f"Invalid input for effect execution: {{e!s}}",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                message=f"Invalid input: {{e!s}}",
                details={{"original_error": str(e)}},
            ) from e

        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            emit_log_event(
                LogLevel.CRITICAL,
                f"Unexpected error during effect execution: {{e!s}}",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"Unexpected error during effect execution: {{e!s}}",
                details={{"original_error": str(e), "error_type": type(e).__name__}},
            ) from e

    def get_metadata_loader(self) -> Any:
        """
        Get metadata loader for this node.

        Returns:
            Metadata loader instance or None if not applicable
        """
        return None

    async def startup(self) -> None:
        """
        Node startup lifecycle hook.

        Publishes introspection data to registry and starts background tasks.
        Should be called when node is ready to serve requests.
        """
        if not MIXINS_AVAILABLE:
            emit_log_event(
                LogLevel.WARNING,
                "Mixins not available - skipping startup registration",
                {{"node_id": str(self.node_id)}}
            )
            return

        emit_log_event(
            LogLevel.INFO,
            "{context['node_class_name']} starting up",
            {{"node_id": str(self.node_id)}}
        )

        # Publish introspection broadcast to registry
        await self.publish_introspection(reason="startup")

        # Start introspection background tasks (heartbeat, registry listener)
        await self.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=30,
            enable_registry_listener=True,
        )

        emit_log_event(
            LogLevel.INFO,
            "{context['node_class_name']} startup complete - node registered",
            {{"node_id": str(self.node_id)}}
        )

    async def shutdown(self) -> None:
        """
        Node shutdown lifecycle hook.

        Stops background tasks and cleans up resources.
        Should be called when node is preparing to exit.
        """
        if not MIXINS_AVAILABLE:
            return

        emit_log_event(
            LogLevel.INFO,
            "{context['node_class_name']} shutting down",
            {{"node_id": str(self.node_id)}}
        )

        # Stop introspection background tasks
        await self.stop_introspection_tasks()

        emit_log_event(
            LogLevel.INFO,
            "{context['node_class_name']} shutdown complete",
            {{"node_id": str(self.node_id)}}
        )


__all__ = ["{context['node_class_name']}"]
'''

    def _get_compute_template(self, context: dict[str, Any]) -> str:
        """Inline template for Compute nodes with introspection and registration."""
        # Generate dependency imports
        dependency_imports = self._generate_dependency_imports(
            context.get("dependencies", {})
        )

        return f'''#!/usr/bin/env python3
"""
{context['node_class_name']} - {context['business_description']}

ONEX v2.0 Compute Node with Registration & Introspection
Domain: {context['domain']}
Generated: {context['generated_at']}
"""

import os
from typing import Any, Dict, List, Optional

from omnibase_core import ModelOnexError, EnumCoreErrorCode
from omnibase_core.nodes.node_compute import NodeCompute
from omnibase_core.models.contracts.model_contract_compute import ModelContractCompute
from omnibase_core.models.core import ModelContainer
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event

# Import mixins from omnibase_core (ONEX standard)
from omnibase_core.mixins.mixin_introspection import MixinNodeIntrospection
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck

{dependency_imports}

MIXINS_AVAILABLE = True


class {context['node_class_name']}(NodeCompute, MixinHealthCheck, MixinNodeIntrospection):
    """
    {context['business_description']}

    Pure transformation/computation logic with automatic registration.
    """

    def __init__(self, container: ModelContainer) -> None:
        """Initialize {context['node_class_name']} with registration and introspection."""
        super().__init__(container)

        # Configuration
        self.config = container.value if isinstance(container.value, dict) else {{}}

        # Initialize health checks and introspection
        if MIXINS_AVAILABLE:
            self.initialize_health_checks()
            self.initialize_introspection()

        emit_log_event(
            LogLevel.INFO,
            "{context['node_class_name']} initialized with registration support",
            {{"node_id": str(self.node_id), "mixins_available": MIXINS_AVAILABLE}}
        )

    async def execute_compute(self, contract: ModelContractCompute) -> Any:
        """
        Execute pure computation.

        Args:
            contract: Compute contract with input data

        Returns:
            Transformed/computed result

        Raises:
            ModelOnexError: If computation fails
        """
        emit_log_event(
            LogLevel.INFO,
            "Executing computation",
            {{
                "node_id": str(self.node_id),
                "correlation_id": str(contract.correlation_id),
            }},
        )

        try:
            # IMPLEMENTATION REQUIRED: Add pure computation logic here
            result = {{"status": "success", "computed": True}}

            emit_log_event(
                LogLevel.INFO,
                "Computation executed successfully",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                }},
            )

            return result

        except (ValueError, TypeError, KeyError) as e:
            # Data validation or type errors
            emit_log_event(
                LogLevel.ERROR,
                f"Data validation error during computation: {{e!s}}",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                    "error_type": type(e).__name__,
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                message=f"Data validation error: {{e!s}}",
                details={{"original_error": str(e), "error_type": type(e).__name__}},
            ) from e

        except (ArithmeticError, OverflowError, ZeroDivisionError) as e:
            # Mathematical computation errors
            emit_log_event(
                LogLevel.ERROR,
                f"Mathematical error during computation: {{e!s}}",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"Mathematical error: {{e!s}}",
                details={{"original_error": str(e)}},
            ) from e

        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            emit_log_event(
                LogLevel.CRITICAL,
                f"Unexpected error during computation: {{e!s}}",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"Unexpected error during computation: {{e!s}}",
                details={{"original_error": str(e), "error_type": type(e).__name__}},
            ) from e

    def get_metadata_loader(self) -> Any:
        """
        Get metadata loader for this node.

        Returns:
            Metadata loader instance or None if not applicable
        """
        return None

    async def startup(self) -> None:
        """Node startup lifecycle hook - publishes introspection and starts background tasks."""
        if not MIXINS_AVAILABLE:
            return
        await self.publish_introspection(reason="startup")
        await self.start_introspection_tasks(enable_heartbeat=True, heartbeat_interval_seconds=30)

    async def shutdown(self) -> None:
        """Node shutdown lifecycle hook - stops background tasks."""
        if not MIXINS_AVAILABLE:
            return
        await self.stop_introspection_tasks()


__all__ = ["{context['node_class_name']}"]
'''

    def _get_reducer_template(self, context: dict[str, Any]) -> str:
        """Inline template for Reducer nodes with introspection and registration."""
        # Generate dependency imports
        dependency_imports = self._generate_dependency_imports(
            context.get("dependencies", {})
        )

        return f'''#!/usr/bin/env python3
"""
{context['node_class_name']} - {context['business_description']}

ONEX v2.0 Reducer Node with Registration & Introspection
Domain: {context['domain']}
Generated: {context['generated_at']}
"""

import os
from typing import Any, Dict, List, Optional

from omnibase_core import ModelOnexError, EnumCoreErrorCode
from omnibase_core.nodes.node_reducer import NodeReducer
from omnibase_core.models.contracts.model_contract_reducer import ModelContractReducer
from omnibase_core.models.core import ModelContainer
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event

# Import mixins from omnibase_core (ONEX standard)
from omnibase_core.mixins.mixin_introspection import MixinNodeIntrospection
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck

{dependency_imports}

MIXINS_AVAILABLE = True


class {context['node_class_name']}(NodeReducer, MixinHealthCheck, MixinNodeIntrospection):
    """
    {context['business_description']}

    Aggregates and reduces data streams with automatic registration.
    """

    def __init__(self, container: ModelContainer) -> None:
        """Initialize {context['node_class_name']} with registration and introspection."""
        super().__init__(container)

        # Configuration
        self.config = container.value if isinstance(container.value, dict) else {{}}
        self.accumulated_state: dict[str, Any] = {{}}

        # Initialize health checks and introspection
        if MIXINS_AVAILABLE:
            self.initialize_health_checks()
            self.initialize_introspection()

        emit_log_event(
            LogLevel.INFO,
            "{context['node_class_name']} initialized with registration support",
            {{"node_id": str(self.node_id), "mixins_available": MIXINS_AVAILABLE}}
        )

    async def execute_reduction(self, contract: ModelContractReducer) -> Any:
        """
        Execute reduction/aggregation.

        Args:
            contract: Reducer contract with data to aggregate

        Returns:
            Aggregated result

        Raises:
            ModelOnexError: If reduction fails
        """
        emit_log_event(
            LogLevel.INFO,
            "Executing reduction",
            {{
                "node_id": str(self.node_id),
                "correlation_id": str(contract.correlation_id),
            }},
        )

        try:
            # IMPLEMENTATION REQUIRED: Add aggregation/reduction logic here
            # Update accumulated_state with new data
            result = {{"status": "success", "aggregated": True}}

            emit_log_event(
                LogLevel.INFO,
                "Reduction executed successfully",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                }},
            )

            return result

        except (ValueError, TypeError, KeyError) as e:
            # Data validation or access errors
            emit_log_event(
                LogLevel.ERROR,
                f"Data validation error during reduction: {{e!s}}",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                    "error_type": type(e).__name__,
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                message=f"Data validation error: {{e!s}}",
                details={{"original_error": str(e), "error_type": type(e).__name__}},
            ) from e

        except (MemoryError, OverflowError) as e:
            # Resource exhaustion during aggregation
            emit_log_event(
                LogLevel.ERROR,
                f"Resource exhaustion during reduction: {{e!s}}",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.RESOURCE_EXHAUSTED,
                message=f"Resource exhaustion: {{e!s}}",
                details={{"original_error": str(e)}},
            ) from e

        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            emit_log_event(
                LogLevel.CRITICAL,
                f"Unexpected error during reduction: {{e!s}}",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"Unexpected error during reduction: {{e!s}}",
                details={{"original_error": str(e), "error_type": type(e).__name__}},
            ) from e

    def get_metadata_loader(self) -> Any:
        """
        Get metadata loader for this node.

        Returns:
            Metadata loader instance or None if not applicable
        """
        return None

    async def startup(self) -> None:
        """Node startup lifecycle hook - publishes introspection and starts background tasks."""
        if not MIXINS_AVAILABLE:
            return
        await self.publish_introspection(reason="startup")
        await self.start_introspection_tasks(enable_heartbeat=True, heartbeat_interval_seconds=30)

    async def shutdown(self) -> None:
        """Node shutdown lifecycle hook - stops background tasks."""
        if not MIXINS_AVAILABLE:
            return
        await self.stop_introspection_tasks()


__all__ = ["{context['node_class_name']}"]
'''

    def _get_orchestrator_template(self, context: dict[str, Any]) -> str:
        """Inline template for Orchestrator nodes with introspection and registration."""
        # Generate dependency imports
        dependency_imports = self._generate_dependency_imports(
            context.get("dependencies", {})
        )

        return f'''#!/usr/bin/env python3
"""
{context['node_class_name']} - {context['business_description']}

ONEX v2.0 Orchestrator Node with Registration & Introspection
Domain: {context['domain']}
Generated: {context['generated_at']}
"""

import os
from typing import Any, Dict, List, Optional

from omnibase_core import ModelOnexError, EnumCoreErrorCode
from omnibase_core.nodes.node_orchestrator import NodeOrchestrator
from omnibase_core.models.contracts.model_contract_orchestrator import ModelContractOrchestrator
from omnibase_core.models.core import ModelContainer
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event

# Import mixins from omnibase_core (ONEX standard)
from omnibase_core.mixins.mixin_introspection import MixinNodeIntrospection
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck

{dependency_imports}

MIXINS_AVAILABLE = True


class {context['node_class_name']}(NodeOrchestrator, MixinHealthCheck, MixinNodeIntrospection):
    """
    {context['business_description']}

    Coordinates workflow execution with automatic registration.
    """

    def __init__(self, container: ModelContainer) -> None:
        """Initialize {context['node_class_name']} with registration and introspection."""
        super().__init__(container)

        # Configuration
        self.config = container.value if isinstance(container.value, dict) else {{}}

        # Initialize health checks and introspection
        if MIXINS_AVAILABLE:
            self.initialize_health_checks()
            self.initialize_introspection()

        emit_log_event(
            LogLevel.INFO,
            "{context['node_class_name']} initialized with registration support",
            {{"node_id": str(self.node_id), "mixins_available": MIXINS_AVAILABLE}}
        )

    async def execute_orchestration(self, contract: ModelContractOrchestrator) -> Any:
        """
        Execute orchestration workflow.

        Args:
            contract: Orchestrator contract with workflow parameters

        Returns:
            Workflow execution result

        Raises:
            ModelOnexError: If orchestration fails
        """
        emit_log_event(
            LogLevel.INFO,
            "Executing orchestration",
            {{
                "node_id": str(self.node_id),
                "correlation_id": str(contract.correlation_id),
            }},
        )

        try:
            # IMPLEMENTATION REQUIRED: Add workflow orchestration logic here
            result = {{"status": "success", "workflow_completed": True}}

            emit_log_event(
                LogLevel.INFO,
                "Orchestration executed successfully",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                }},
            )

            return result

        except (ConnectionError, TimeoutError) as e:
            # Network errors calling downstream nodes
            emit_log_event(
                LogLevel.ERROR,
                f"Network error during orchestration: {{e!s}}",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                    "error_type": type(e).__name__,
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.NETWORK_ERROR,
                message=f"Network error in workflow: {{e!s}}",
                details={{"original_error": str(e), "error_type": type(e).__name__}},
            ) from e

        except ValueError as e:
            # Invalid workflow configuration
            emit_log_event(
                LogLevel.ERROR,
                f"Invalid workflow configuration: {{e!s}}",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                message=f"Invalid workflow configuration: {{e!s}}",
                details={{"original_error": str(e)}},
            ) from e

        except Exception as e:
            # Unexpected errors - log with full traceback and re-raise
            emit_log_event(
                LogLevel.CRITICAL,
                f"Unexpected error during orchestration: {{e!s}}",
                {{
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"Unexpected error during orchestration: {{e!s}}",
                details={{"original_error": str(e), "error_type": type(e).__name__}},
            ) from e

    def get_metadata_loader(self) -> Any:
        """
        Get metadata loader for this node.

        Returns:
            Metadata loader instance or None if not applicable
        """
        return None

    async def startup(self) -> None:
        """Node startup lifecycle hook - publishes introspection and starts background tasks."""
        if not MIXINS_AVAILABLE:
            return
        await self.publish_introspection(reason="startup")
        await self.start_introspection_tasks(enable_heartbeat=True, heartbeat_interval_seconds=30)

    async def shutdown(self) -> None:
        """Node shutdown lifecycle hook - stops background tasks."""
        if not MIXINS_AVAILABLE:
            return
        await self.stop_introspection_tasks()


__all__ = ["{context['node_class_name']}"]
'''

    def _get_io_operations_section(
        self, node_type: EnumNodeType, context: dict[str, Any]
    ) -> str:
        """
        Generate io_operations section for Effect nodes only.

        Effect nodes require io_operations field per omnibase_core ModelIOOperationConfig.
        Other node types (Compute, Reducer, Orchestrator) do not.

        ModelIOOperationConfig requires:
        - operation_type (string, required): Type of I/O operation
        - atomic (bool, default=true): Whether operation should be atomic
        - backup_enabled (bool, default=false): Enable backup before destructive operations
        - permissions (string, optional): File permissions or access rights
        - recursive (bool, default=false): Enable recursive operations
        - buffer_size (int, default=8192): Buffer size for streaming
        - timeout_seconds (int, default=30): Operation timeout
        - validation_enabled (bool, default=true): Enable result validation
        """
        if node_type != EnumNodeType.EFFECT:
            return ""

        # Generate io_operations section for Effect nodes using ModelIOOperationConfig format
        return """# IO operations for EFFECT node (required)
# Each operation follows ModelIOOperationConfig schema
io_operations:
  - operation_type: "database_query"
    atomic: true
    timeout_seconds: 30
    validation_enabled: true

"""

    def _get_inline_contract_template(
        self, node_type: EnumNodeType, context: dict[str, Any]
    ) -> str:
        """
        Get inline contract template in NEW omnibase_core format.

        Required fields (11 total):
        1. name (string)
        2. version (dict with major/minor/patch)
        3. description (string)
        4. node_type (UPPERCASE string)
        5. input_model (string - model class name)
        6. output_model (string - model class name)
        7. tool_specification (dict with main_tool_class)
        8-11. Legacy compatibility fields (node_name, node_version, contract_name, author)
        """
        return f"""# {context['node_class_name']} Contract - ONEX v2.0 Compliant
# Generated: {context['generated_at'].split('T')[0]}

# Required fields for ModelContract{node_type.value.capitalize()}
name: "{context['service_name']}"
version:
  major: {context['version_dict']['major']}
  minor: {context['version_dict']['minor']}
  patch: {context['version_dict']['patch']}
description: "{context['business_description']}"
node_type: "{context['node_type_upper']}"

# Input/Output Models (required by omnibase_core) - must be strings (model class names)
input_model: "{context['input_model']}"
output_model: "{context['output_model']}"

# State schemas (required by ONEX v2.0)
input_state:
  $ref: "contracts/{context['service_name']}-input-state.yaml"
output_state:
  $ref: "contracts/{context['service_name']}-output-state.yaml"

# Tool specification (required by omnibase_core)
tool_specification:
  tool_name: "{context['service_name']}"
  main_tool_class: "{context['package_path']}.{context['node_class_name']}"

{self._get_io_operations_section(node_type, context)}
# Legacy fields for compatibility
node_name: {context['service_name']}
node_version:
  major: {context['version_dict']['major']}
  minor: {context['version_dict']['minor']}
  patch: {context['version_dict']['patch']}

contract_name: {context['service_name']}_contract
author: OmniNode Code Generation
created_at: "{context['generated_at'].split('T')[0]}"

# Core Metadata
metadata:
  domain: "{context['domain']}"
  generated: true
  generated_at: "{context['generated_at']}"
  tags:
{chr(10).join(f'    - "{feature}"' for feature in context['features'])}

# Operations supported by this node
operations:
{chr(10).join(f'  - "{op}"' for op in context['operations'])}

# Performance Requirements
performance_requirements:
  execution_time:
    target_ms: 100
    max_ms: 1000
  memory_usage:
    target_mb: 128
    max_mb: 512

# Testing Requirements
testing:
  unit_tests:
    coverage_target: 85
    required: true
  integration_tests:
    required: true
"""

    def _get_request_model_template(self, context: dict[str, Any]) -> str:
        """Generate request model template."""
        return f'''#!/usr/bin/env python3
"""Request models for {context['service_name']}."""

from pydantic import BaseModel, Field


class Model{context['pascal_name']}Request(BaseModel):
    """Request model for {context['service_name']} operations."""

    # SCHEMA REQUIRED: Add request-specific fields below
    operation: str = Field(..., description="Operation to perform")


__all__ = ["Model{context['pascal_name']}Request"]
'''

    def _get_response_model_template(self, context: dict[str, Any]) -> str:
        """Generate response model template."""
        return f'''#!/usr/bin/env python3
"""Response models for {context['service_name']}."""

from pydantic import BaseModel, Field


class Model{context['pascal_name']}Response(BaseModel):
    """Response model for {context['service_name']} operations."""

    # SCHEMA REQUIRED: Add response-specific fields below
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")


__all__ = ["Model{context['pascal_name']}Response"]
'''

    def _get_unit_test_template(
        self, node_type: EnumNodeType, context: dict[str, Any]
    ) -> str:
        """Generate unit test template."""
        return f'''#!/usr/bin/env python3
"""Unit tests for {context['node_class_name']}."""

import pytest
from omnibase_core.models.core import ModelContainer
from omnibase_core.models.contracts.model_contract_{node_type.value} import ModelContract{node_type.value.capitalize()}

from ..node import {context['node_class_name']}


@pytest.fixture
def container():
    """Create test container."""
    return ModelContainer()


@pytest.fixture
def node(container):
    """Create test node instance."""
    return {context['node_class_name']}(container)


@pytest.mark.asyncio
async def test_node_initialization(node):
    """Test node initializes correctly."""
    assert node is not None
    assert node.node_id is not None


@pytest.mark.asyncio
async def test_execute_{node_type.value}(node):
    """Test {node_type.value} execution."""
    contract = ModelContract{node_type.value.capitalize()}(
        # CONTRACT CONFIGURATION: Add node-specific contract parameters
    )

    result = await node.execute_{node_type.value}(contract)

    assert result is not None
    # TEST IMPLEMENTATION: Add unit test assertions
'''

    def _get_integration_test_template(self, context: dict[str, Any]) -> str:
        """Generate integration test template."""
        return f'''#!/usr/bin/env python3
"""Integration tests for {context['node_class_name']}."""

import pytest
from omnibase_core.models.core import ModelContainer

from ..node import {context['node_class_name']}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test end-to-end workflow."""
    # INTEGRATION TEST: Implement end-to-end workflow testing
    pass
'''

    def _write_artifacts_to_disk(
        self, artifacts: ModelGeneratedArtifacts, output_directory: Path
    ) -> None:
        """
        Write generated artifacts to disk.

        Args:
            artifacts: Generated code artifacts
            output_directory: Target directory

        Note:
            This is required for test execution via pytest.
        """
        logger.debug(f"Writing artifacts to {output_directory}")

        all_files = artifacts.get_all_files()
        for filename, content in all_files.items():
            file_path = output_directory / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

        logger.info(f"Wrote {len(all_files)} files to {output_directory}")


# Export
__all__ = ["TemplateEngine", "ModelGeneratedArtifacts"]
