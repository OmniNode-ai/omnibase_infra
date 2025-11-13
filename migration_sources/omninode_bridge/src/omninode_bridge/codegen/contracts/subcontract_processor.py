#!/usr/bin/env python3
"""
Subcontract Processor for ONEX v2.0 Code Generation (Phase 3).

Processes subcontract references in contracts and generates appropriate code snippets.
Supports 6 subcontract types:
- database: Database operations (CRUD, queries)
- api: API client operations
- event: Event handling (Kafka, pub/sub)
- compute: Computation logic
- state: State management
- workflow: Workflow coordination

Features:
- Code generation from subcontract definitions
- Dependency tracking for imports
- Template-based code generation
- Nested subcontract support

Thread-safe and stateless.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, Template

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Models
# ============================================================================


class EnumSubcontractType(str, Enum):
    """
    Subcontract type enumeration (Phase 3).

    Defines the 6 supported subcontract types for code generation.
    """

    DATABASE = "database"  # Database operations (CRUD, queries)
    API = "api"  # API client operations
    EVENT = "event"  # Event handling (Kafka, pub/sub)
    COMPUTE = "compute"  # Computation logic
    STATE = "state"  # State management
    WORKFLOW = "workflow"  # Workflow coordination


@dataclass
class ModelProcessedSubcontract:
    """
    Represents a processed subcontract with generated code.

    Attributes:
        subcontract_type: Type of subcontract
        name: Subcontract name/identifier
        code: Generated code snippet
        imports: Required imports for this subcontract
        dependencies: Dependencies on other subcontracts
        metadata: Additional metadata (templates used, etc.)
    """

    subcontract_type: EnumSubcontractType
    name: str
    code: str = ""
    imports: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSubcontractResults:
    """
    Results of subcontract processing.

    Attributes:
        processed_subcontracts: List of processed subcontracts
        all_imports: Deduplicated list of all required imports
        dependency_graph: Map of subcontract dependencies
        errors: Processing errors
        warnings: Processing warnings
    """

    processed_subcontracts: list[ModelProcessedSubcontract] = field(
        default_factory=list
    )
    all_imports: list[str] = field(default_factory=list)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if processing had errors."""
        return len(self.errors) > 0

    @property
    def success_count(self) -> int:
        """Count of successfully processed subcontracts."""
        return len(self.processed_subcontracts)


# ============================================================================
# Subcontract Processor
# ============================================================================


class SubcontractProcessor:
    """
    Process subcontract references and generate code.

    Provides comprehensive subcontract processing with:
    - Template-based code generation
    - Dependency tracking
    - Import management
    - Validation

    Thread-safe and stateless - can be reused across calls.

    Example:
        >>> processor = SubcontractProcessor()
        >>> contract = ... # ModelEnhancedContract
        >>> results = processor.process_subcontracts(contract)
        >>> if not results.has_errors:
        ...     print(f"Processed {results.success_count} subcontracts")
        ...     print(f"Required imports: {results.all_imports}")
    """

    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize subcontract processor.

        Args:
            template_dir: Directory containing subcontract templates
                         (defaults to templates/subcontracts in package)
        """
        if template_dir is None:
            # Default to templates/subcontracts directory in package
            template_dir = Path(__file__).parent.parent / "templates" / "subcontracts"

        self.template_dir = template_dir
        self._load_templates()

    def _load_templates(self) -> None:
        """Load Jinja2 templates for subcontracts."""
        try:
            # Create Jinja2 environment
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )

            # Load templates for each subcontract type
            self.templates: dict[EnumSubcontractType, Template] = {}

            for subcontract_type in EnumSubcontractType:
                template_name = f"{subcontract_type.value}_subcontract.j2"
                try:
                    template = self.jinja_env.get_template(template_name)
                    self.templates[subcontract_type] = template
                    logger.debug(f"Loaded template: {template_name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to load template {template_name}: {e}. "
                        f"Will use fallback for {subcontract_type.value} subcontracts."
                    )

        except Exception as e:
            logger.error(f"Failed to initialize Jinja2 environment: {e}")
            # Create empty template dict to allow fallback behavior
            self.templates = {}

    def process_subcontracts(
        self,
        contract: Any,  # ModelEnhancedContract type hint would be circular
    ) -> ModelSubcontractResults:
        """
        Process all subcontracts in contract.

        Args:
            contract: ModelEnhancedContract instance

        Returns:
            ModelSubcontractResults with processed subcontracts and metadata

        Example:
            >>> results = processor.process_subcontracts(contract)
            >>> for sub in results.processed_subcontracts:
            ...     print(f"{sub.name}: {len(sub.code)} chars")
        """
        results = ModelSubcontractResults()

        # Check if contract has subcontracts
        if not contract.subcontracts:
            logger.debug(f"Contract {contract.name} has no subcontracts")
            return results

        # Process each subcontract
        for subcontract_name, subcontract_data in contract.subcontracts.items():
            if not isinstance(subcontract_data, dict):
                error_msg = (
                    f"Invalid subcontract '{subcontract_name}': "
                    f"must be dict, got {type(subcontract_data).__name__}"
                )
                results.errors.append(error_msg)
                logger.error(error_msg)
                continue

            # Determine subcontract type
            subcontract_type_str = subcontract_data.get("type", "")
            try:
                subcontract_type = EnumSubcontractType(subcontract_type_str)
            except ValueError:
                error_msg = (
                    f"Invalid subcontract type '{subcontract_type_str}' for '{subcontract_name}'. "
                    f"Must be one of: {', '.join(t.value for t in EnumSubcontractType)}"
                )
                results.errors.append(error_msg)
                logger.error(error_msg)
                continue

            # Process subcontract based on type
            processed = self._process_subcontract_by_type(
                subcontract_type, subcontract_name, subcontract_data, results
            )

            if processed:
                results.processed_subcontracts.append(processed)
                # Track dependencies
                self._track_dependencies(processed, results)

        # Deduplicate and sort imports
        self._consolidate_imports(results)

        logger.info(
            f"Processed {results.success_count} subcontracts for contract {contract.name} "
            f"({len(results.errors)} errors, {len(results.warnings)} warnings)"
        )

        return results

    def _process_subcontract_by_type(
        self,
        subcontract_type: EnumSubcontractType,
        name: str,
        data: dict[str, Any],
        results: ModelSubcontractResults,
    ) -> Optional[ModelProcessedSubcontract]:
        """
        Process subcontract based on its type.

        Args:
            subcontract_type: Type of subcontract
            name: Subcontract name
            data: Subcontract configuration data
            results: Results accumulator (for errors/warnings)

        Returns:
            ModelProcessedSubcontract or None if processing failed
        """
        # Dispatch to appropriate processor
        processors = {
            EnumSubcontractType.DATABASE: self._process_database_subcontract,
            EnumSubcontractType.API: self._process_api_subcontract,
            EnumSubcontractType.EVENT: self._process_event_subcontract,
            EnumSubcontractType.COMPUTE: self._process_compute_subcontract,
            EnumSubcontractType.STATE: self._process_state_subcontract,
            EnumSubcontractType.WORKFLOW: self._process_workflow_subcontract,
        }

        processor_func = processors.get(subcontract_type)
        if not processor_func:
            error_msg = f"No processor for subcontract type: {subcontract_type}"
            results.errors.append(error_msg)
            logger.error(error_msg)
            return None

        try:
            return processor_func(name, data)
        except Exception as e:
            error_msg = (
                f"Failed to process {subcontract_type.value} subcontract '{name}': {e}"
            )
            results.errors.append(error_msg)
            logger.error(error_msg)
            return None

    def _process_database_subcontract(
        self, name: str, data: dict[str, Any]
    ) -> ModelProcessedSubcontract:
        """Generate code for database subcontract."""
        # Render template or generate fallback code
        code = self._render_template(EnumSubcontractType.DATABASE, data)

        imports = [
            "from omnibase_core.database import DatabaseClient",
            "from typing import Any, Optional",
        ]

        # Check for specific database features
        if data.get("use_transactions", False):
            imports.append("from omnibase_core.database import Transaction")

        if data.get("use_connection_pool", True):
            imports.append("from omnibase_core.database import ConnectionPool")

        return ModelProcessedSubcontract(
            subcontract_type=EnumSubcontractType.DATABASE,
            name=name,
            code=code,
            imports=imports,
            metadata={"operations": data.get("operations", [])},
        )

    def _process_api_subcontract(
        self, name: str, data: dict[str, Any]
    ) -> ModelProcessedSubcontract:
        """Generate code for API client subcontract."""
        code = self._render_template(EnumSubcontractType.API, data)

        imports = [
            "import httpx",
            "from typing import Any, Optional",
        ]

        # Check for specific API features
        if data.get("use_auth", False):
            imports.append("from omnibase_core.auth import AuthProvider")

        if data.get("use_retry", True):
            imports.append("from omnibase_core.retry import RetryPolicy")

        return ModelProcessedSubcontract(
            subcontract_type=EnumSubcontractType.API,
            name=name,
            code=code,
            imports=imports,
            metadata={"endpoints": data.get("endpoints", [])},
        )

    def _process_event_subcontract(
        self, name: str, data: dict[str, Any]
    ) -> ModelProcessedSubcontract:
        """Generate code for event handling subcontract."""
        code = self._render_template(EnumSubcontractType.EVENT, data)

        imports = [
            "from omnibase_core.events import EventPublisher, EventConsumer",
            "from omnibase_core.models.events import OnexEnvelopeV1",
        ]

        # Check for Kafka-specific features
        if data.get("use_kafka", True):
            imports.append("from aiokafka import AIOKafkaProducer, AIOKafkaConsumer")

        return ModelProcessedSubcontract(
            subcontract_type=EnumSubcontractType.EVENT,
            name=name,
            code=code,
            imports=imports,
            metadata={"topics": data.get("topics", [])},
        )

    def _process_compute_subcontract(
        self, name: str, data: dict[str, Any]
    ) -> ModelProcessedSubcontract:
        """Generate code for computation subcontract."""
        code = self._render_template(EnumSubcontractType.COMPUTE, data)

        imports = [
            "from typing import Any",
        ]

        # Check for async computation
        if data.get("async_compute", True):
            imports.append("import asyncio")

        return ModelProcessedSubcontract(
            subcontract_type=EnumSubcontractType.COMPUTE,
            name=name,
            code=code,
            imports=imports,
            metadata={"algorithm": data.get("algorithm", "custom")},
        )

    def _process_state_subcontract(
        self, name: str, data: dict[str, Any]
    ) -> ModelProcessedSubcontract:
        """Generate code for state management subcontract."""
        code = self._render_template(EnumSubcontractType.STATE, data)

        imports = [
            "from omnibase_core.state import StateManager",
            "from typing import Any, Optional",
        ]

        # Check for FSM
        if data.get("use_fsm", False):
            imports.append("from omnibase_core.fsm import FiniteStateMachine")

        return ModelProcessedSubcontract(
            subcontract_type=EnumSubcontractType.STATE,
            name=name,
            code=code,
            imports=imports,
            metadata={"states": data.get("states", [])},
        )

    def _process_workflow_subcontract(
        self, name: str, data: dict[str, Any]
    ) -> ModelProcessedSubcontract:
        """Generate code for workflow coordination subcontract."""
        code = self._render_template(EnumSubcontractType.WORKFLOW, data)

        imports = [
            "from omnibase_core.workflow import WorkflowOrchestrator",
            "from typing import Any, Optional",
        ]

        # Check for specific workflow features
        if data.get("use_llama_index", False):
            imports.append(
                "from llama_index.core.workflow import Workflow, StartEvent, StopEvent"
            )

        return ModelProcessedSubcontract(
            subcontract_type=EnumSubcontractType.WORKFLOW,
            name=name,
            code=code,
            imports=imports,
            metadata={"steps": data.get("steps", [])},
        )

    def _render_template(
        self, subcontract_type: EnumSubcontractType, data: dict[str, Any]
    ) -> str:
        """
        Render Jinja2 template for subcontract.

        Args:
            subcontract_type: Type of subcontract
            data: Template context data

        Returns:
            Rendered code string
        """
        template = self.templates.get(subcontract_type)

        if template:
            try:
                return template.render(**data)
            except Exception as e:
                logger.warning(
                    f"Template rendering failed for {subcontract_type.value}: {e}. "
                    f"Using fallback."
                )

        # Fallback: generate simple TODO comment
        return f"""
    # TODO: Implement {subcontract_type.value} subcontract logic
    # Configuration: {data}
    pass
"""

    def _track_dependencies(
        self, subcontract: ModelProcessedSubcontract, results: ModelSubcontractResults
    ) -> None:
        """
        Track dependencies for a processed subcontract.

        Args:
            subcontract: Processed subcontract
            results: Results accumulator
        """
        # Add dependencies to dependency graph
        if subcontract.dependencies:
            results.dependency_graph[subcontract.name] = subcontract.dependencies

        # Add imports to results
        for imp in subcontract.imports:
            if imp not in results.all_imports:
                results.all_imports.append(imp)

    def _consolidate_imports(self, results: ModelSubcontractResults) -> None:
        """
        Deduplicate and sort imports.

        Args:
            results: Results accumulator
        """
        # Deduplicate
        unique_imports = list(set(results.all_imports))

        # Sort: standard library first, then third-party, then omnibase_core
        def import_sort_key(imp: str) -> tuple[int, str]:
            if imp.startswith("from omnibase_core") or imp.startswith(
                "import omnibase_core"
            ):
                return (2, imp)
            elif any(
                imp.startswith(f"from {lib}") or imp.startswith(f"import {lib}")
                for lib in ["typing", "asyncio", "dataclasses", "enum"]
            ):
                return (0, imp)
            else:
                return (1, imp)

        results.all_imports = sorted(unique_imports, key=import_sort_key)


# Export
__all__ = [
    "EnumSubcontractType",
    "ModelProcessedSubcontract",
    "ModelSubcontractResults",
    "SubcontractProcessor",
]
