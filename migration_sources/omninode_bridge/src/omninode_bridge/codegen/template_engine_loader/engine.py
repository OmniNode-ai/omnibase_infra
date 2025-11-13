#!/usr/bin/env python3
"""
Template Engine for loading ONEX node templates.

Discovers and loads pre-written Python template files from directory structure,
detects stubs for LLM enhancement, and prepares artifacts for BusinessLogicGenerator.

This is an alternative to Jinja2-based generation - for cases where templates
are hand-written Python files rather than generated from Jinja2.

ONEX v2.0 Compliance:
- Type-safe template loading
- Structured error handling with ModelOnexError
- Comprehensive metadata extraction
- Integration with CodeInjector for stub detection

Example Usage:
    >>> from pathlib import Path
    >>> engine = TemplateEngine()
    >>>
    >>> # Discover all templates
    >>> templates = engine.discover_templates()
    >>> print(f"Found {len(templates)} templates")
    >>>
    >>> # Load specific template
    >>> artifacts = engine.load_template(node_type="effect", version="v1_0_0")
    >>> print(f"Loaded template with {artifacts.get_stub_count()} stubs")
"""

import logging
import re
from pathlib import Path
from typing import Optional

from omnibase_core import EnumCoreErrorCode, ModelOnexError

from ..business_logic.injector import CodeInjector
from .config import TemplateEngineConfig
from .models import (
    ModelStubInfo,
    ModelTemplateArtifacts,
    ModelTemplateInfo,
    ModelTemplateMetadata,
)

logger = logging.getLogger(__name__)


class TemplateEngineError(Exception):
    """Raised when template operations fail."""

    pass


class TemplateEngine:
    """
    Loads and manages ONEX node templates.

    Discovers templates in directory structure, loads template files,
    detects stubs, and prepares artifacts for BusinessLogicGenerator.

    Features:
    - Template discovery by node type and version
    - Automatic stub detection using CodeInjector
    - Metadata extraction from template files
    - Validation of required methods
    """

    def __init__(
        self,
        template_root: Optional[Path] = None,
        enable_validation: bool = True,
    ):
        """
        Initialize TemplateEngine.

        Args:
            template_root: Root directory for templates (defaults to config value)
            enable_validation: Enable template validation
        """
        self.template_root = template_root or TemplateEngineConfig.DEFAULT_TEMPLATE_ROOT
        self.enable_validation = enable_validation

        # Initialize CodeInjector for stub detection
        self.code_injector = CodeInjector()

        logger.info(
            f"TemplateEngine initialized with template_root={self.template_root}, "
            f"validation={enable_validation}"
        )

    def discover_templates(
        self, directory: Optional[Path] = None
    ) -> list[ModelTemplateInfo]:
        """
        Discover all templates in directory structure.

        Scans template directory for Python files and extracts metadata.

        Args:
            directory: Directory to scan (defaults to template_root)

        Returns:
            List of TemplateInfo for each discovered template

        Raises:
            TemplateEngineError: If directory doesn't exist or scan fails

        Example:
            >>> engine = TemplateEngine()
            >>> templates = engine.discover_templates()
            >>> for template in templates:
            ...     print(f"{template.node_type}/{template.version}: {template.template_name}")
        """
        scan_dir = directory or self.template_root

        if not scan_dir.exists():
            raise TemplateEngineError(f"Template directory does not exist: {scan_dir}")

        logger.info(f"Discovering templates in {scan_dir}")

        templates = []
        discovered_count = 0

        # Scan each node type directory
        for node_type, dir_name in TemplateEngineConfig.NODE_TYPE_DIRECTORIES.items():
            node_type_dir = scan_dir / dir_name

            if not node_type_dir.exists():
                logger.debug(f"Node type directory not found: {node_type_dir}")
                continue

            # Find all Python template files
            for template_file in node_type_dir.glob(
                f"**/{TemplateEngineConfig.TEMPLATE_FILE_PATTERN}"
            ):
                if not template_file.is_file():
                    continue

                # Skip __pycache__ and test files
                if "__pycache__" in str(template_file) or template_file.name.startswith(
                    "test_"
                ):
                    continue

                try:
                    # Extract version from path (e.g., effect/v1_0_0/node.py)
                    version = self._extract_version_from_path(template_file)

                    # Extract metadata from file
                    metadata = self._extract_metadata(template_file)

                    # Create TemplateInfo
                    template_info = ModelTemplateInfo(
                        template_path=template_file,
                        node_type=node_type,
                        version=version,
                        template_name=template_file.stem,  # e.g., "node"
                        metadata=metadata,
                    )

                    templates.append(template_info)
                    discovered_count += 1

                    logger.debug(
                        f"Discovered template: {node_type}/{version}/{template_file.name}"
                    )

                except Exception as e:
                    logger.warning(f"Failed to process template {template_file}: {e}")
                    continue

        logger.info(f"Discovered {discovered_count} templates")
        return templates

    def load_template(
        self, node_type: str, version: str, template_name: str = "node"
    ) -> ModelTemplateArtifacts:
        """
        Load template and detect stubs.

        Args:
            node_type: Node type (effect/compute/reducer/orchestrator)
            version: Version string (e.g., v1_0_0)
            template_name: Template file name without extension (default: "node")

        Returns:
            ModelTemplateArtifacts with template code, stubs, and metadata

        Raises:
            TemplateEngineError: If template not found or loading fails
            ModelOnexError: If validation fails

        Example:
            >>> engine = TemplateEngine()
            >>> artifacts = engine.load_template("effect", "v1_0_0")
            >>> print(f"Loaded {len(artifacts.stubs)} stubs")
        """
        # Validate node type
        if node_type not in TemplateEngineConfig.NODE_TYPE_DIRECTORIES:
            raise TemplateEngineError(
                f"Invalid node_type '{node_type}'. Must be one of: "
                f"{list(TemplateEngineConfig.NODE_TYPE_DIRECTORIES.keys())}"
            )

        # Validate version format
        if not re.match(TemplateEngineConfig.VERSION_PATTERN, version):
            raise TemplateEngineError(
                f"Invalid version format '{version}'. Expected format: v1_0_0"
            )

        # Build template path
        template_path = self._build_template_path(node_type, version, template_name)

        if not template_path.exists():
            raise TemplateEngineError(
                f"Template not found: {template_path}\n"
                f"(node_type={node_type}, version={version}, name={template_name})"
            )

        logger.info(f"Loading template from {template_path}")

        try:
            # Read template file
            template_code = template_path.read_text(encoding="utf-8")

            # Extract metadata
            metadata = self._extract_metadata(template_path)

            # Detect stubs using CodeInjector
            stubs = self._detect_stubs(template_code, str(template_path))

            logger.info(f"Detected {len(stubs)} stubs in template")

            # Validate template if enabled
            if self.enable_validation:
                self._validate_template(node_type, template_code, stubs)

            # Create artifacts
            artifacts = ModelTemplateArtifacts(
                template_code=template_code,
                template_path=template_path,
                stubs=stubs,
                node_type=node_type,
                version=version,
                metadata=metadata,
            )

            logger.info(
                f"Loaded template successfully: {node_type}/{version}/{template_name} "
                f"({len(stubs)} stubs)"
            )

            return artifacts

        except Exception as e:
            if isinstance(e, TemplateEngineError | ModelOnexError):
                raise

            raise TemplateEngineError(
                f"Failed to load template {template_path}: {e}"
            ) from e

    def _build_template_path(
        self, node_type: str, version: str, template_name: str
    ) -> Path:
        """
        Build full path to template file.

        Args:
            node_type: Node type
            version: Version string
            template_name: Template file name without extension

        Returns:
            Full path to template file
        """
        node_type_dir = TemplateEngineConfig.NODE_TYPE_DIRECTORIES[node_type]
        return self.template_root / node_type_dir / version / f"{template_name}.py"

    def _extract_version_from_path(self, template_path: Path) -> str:
        """
        Extract version string from template path.

        Args:
            template_path: Path to template file

        Returns:
            Version string (e.g., v1_0_0)

        Raises:
            TemplateEngineError: If version cannot be extracted
        """
        # Look for version pattern in path parts
        for part in template_path.parts:
            if re.match(TemplateEngineConfig.VERSION_PATTERN, part):
                return part

        # Default to v1_0_0 if not found
        if TemplateEngineConfig.WARN_ON_MISSING_METADATA:
            logger.warning(f"No version found in path {template_path}, using v1_0_0")

        return "v1_0_0"

    def _extract_metadata(self, template_path: Path) -> ModelTemplateMetadata:
        """
        Extract metadata from template file.

        Parses file content to extract:
        - Description (from module docstring)
        - Author (from # Author: comment)
        - Tags (from # Tags: comment)

        Args:
            template_path: Path to template file

        Returns:
            ModelTemplateMetadata with extracted information
        """
        try:
            content = template_path.read_text(encoding="utf-8")

            # Extract description from first triple-quoted docstring
            description = None
            desc_match = re.search(r'"""([^"]+)"""', content, re.DOTALL)
            if desc_match:
                # Clean up docstring (remove extra whitespace, newlines)
                description = " ".join(desc_match.group(1).split())

            # Extract author
            author = None
            author_match = re.search(
                TemplateEngineConfig.METADATA_MARKERS["author"], content
            )
            if author_match:
                author = author_match.group(1).strip()

            # Extract tags
            tags = []
            tags_match = re.search(
                TemplateEngineConfig.METADATA_MARKERS["tags"], content
            )
            if tags_match:
                tags = [tag.strip() for tag in tags_match.group(1).split(",")]

            # Extract version from path
            version = self._extract_version_from_path(template_path)

            # Determine node type from path
            node_type = "unknown"
            for ntype, dir_name in TemplateEngineConfig.NODE_TYPE_DIRECTORIES.items():
                if dir_name in str(template_path):
                    node_type = ntype
                    break

            return ModelTemplateMetadata(
                node_type=node_type,
                version=version,
                description=description,
                author=author,
                created_at=None,
                tags=tags,
            )

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {template_path}: {e}")
            return ModelTemplateMetadata(
                node_type="unknown",
                version="v1_0_0",
                description=None,
                author=None,
                created_at=None,
                tags=[],
            )

    def _detect_stubs(self, template_code: str, file_path: str) -> list[ModelStubInfo]:
        """
        Detect stubs in template code using CodeInjector.

        Args:
            template_code: Template code to scan
            file_path: Path to template file (for StubInfo)

        Returns:
            List of ModelStubInfo for detected stubs
        """
        # Use CodeInjector to find stubs
        injector_stubs = self.code_injector.find_stubs(template_code, file_path)

        # Convert to ModelStubInfo
        stubs = [
            ModelStubInfo(
                method_name=stub.method_name,
                stub_code=stub.stub_code,
                line_start=stub.line_start,
                line_end=stub.line_end,
                signature=stub.signature,
                docstring=stub.docstring,
            )
            for stub in injector_stubs
        ]

        return stubs

    def _validate_template(
        self, node_type: str, template_code: str, stubs: list[ModelStubInfo]
    ) -> None:
        """
        Validate template has required methods.

        Args:
            node_type: Node type
            template_code: Template code
            stubs: Detected stubs

        Raises:
            ModelOnexError: If validation fails
        """
        required_methods = TemplateEngineConfig.REQUIRED_METHODS.get(node_type, ())

        if not required_methods:
            return  # No validation for this node type

        # Check if required methods exist in template
        missing_methods = []
        for required_method in required_methods:
            # Check both in stub names and in raw template code
            found_in_stubs = any(stub.method_name == required_method for stub in stubs)
            found_in_code = f"def {required_method}" in template_code

            if not (found_in_stubs or found_in_code):
                missing_methods.append(required_method)

        if missing_methods:
            if TemplateEngineConfig.STRICT_VALIDATION:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.INVALID_INPUT,
                    message=f"Template missing required methods: {missing_methods}",
                    details={
                        "node_type": node_type,
                        "missing_methods": missing_methods,
                        "required_methods": required_methods,
                    },
                )
            else:
                logger.warning(
                    f"Template for {node_type} missing methods: {missing_methods}"
                )


__all__ = ["TemplateEngine", "TemplateEngineError"]
