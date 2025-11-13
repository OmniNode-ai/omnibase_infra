"""
Pattern Loader for Code Generation.

This module handles loading production patterns from YAML files and converting
them into ModelPatternMetadata objects. It provides caching and validation
for efficient pattern lookup.

Performance Target: <5ms to load all 21 patterns (with caching)
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

from .models import (
    EnumNodeType,
    EnumPatternCategory,
    ModelPatternExample,
    ModelPatternMetadata,
)

logger = logging.getLogger(__name__)


class PatternLoaderError(Exception):
    """Base exception for pattern loading errors."""


class PatternNotFoundError(PatternLoaderError):
    """Pattern file not found."""


class PatternValidationError(PatternLoaderError):
    """Pattern YAML validation failed."""


class PatternLoader:
    """
    Load production patterns from YAML files.

    This class handles reading pattern YAML files from the patterns directory,
    parsing them into ModelPatternMetadata objects, and caching for performance.

    Attributes:
        patterns_dir: Root directory containing pattern YAML files
        _cache: In-memory cache of loaded patterns
        _registry_cache: Cached registry metadata

    Performance:
        - Cold load (first time): ~20-50ms for 21 patterns
        - Warm load (cached): <1ms
        - Memory overhead: ~100KB for 21 patterns
    """

    def __init__(self, patterns_dir: Optional[Path] = None):
        """
        Initialize pattern loader.

        Args:
            patterns_dir: Root directory for patterns (default: auto-detect)
        """
        if patterns_dir is None:
            # Auto-detect patterns directory relative to this file
            patterns_dir = Path(__file__).parent

        self.patterns_dir = Path(patterns_dir)
        self._cache: dict[str, ModelPatternMetadata] = {}
        self._registry_cache: Optional[dict] = None

        if not self.patterns_dir.exists():
            raise PatternLoaderError(
                f"Patterns directory not found: {self.patterns_dir}"
            )

        logger.debug(f"PatternLoader initialized with dir: {self.patterns_dir}")

    def load_all_patterns(self) -> dict[str, ModelPatternMetadata]:
        """
        Load all patterns from the patterns directory.

        Returns:
            Dictionary mapping pattern_id to ModelPatternMetadata

        Raises:
            PatternLoaderError: If loading fails

        Performance: ~20-50ms cold, <1ms warm (cached)
        """
        if self._cache:
            logger.debug(f"Returning {len(self._cache)} cached patterns")
            return self._cache.copy()

        logger.info("Loading all patterns from disk...")
        registry = self._load_registry()

        patterns = {}
        for pattern_entry in registry.get("patterns", []):
            pattern_name = pattern_entry["name"]
            pattern_file = pattern_entry["file"]

            try:
                pattern = self.load_pattern_by_file(pattern_file)
                patterns[pattern.pattern_id] = pattern
                logger.debug(f"Loaded pattern: {pattern.pattern_id}")
            except Exception as e:
                logger.error(
                    f"Failed to load pattern {pattern_name} from {pattern_file}: {e}"
                )
                # Continue loading other patterns
                continue

        self._cache = patterns
        logger.info(f"Loaded {len(patterns)} patterns successfully")
        return patterns.copy()

    def load_pattern(self, pattern_name: str) -> ModelPatternMetadata:
        """
        Load a single pattern by name.

        Args:
            pattern_name: Name of the pattern (e.g., "error_handling")

        Returns:
            ModelPatternMetadata object

        Raises:
            PatternNotFoundError: If pattern not found
            PatternValidationError: If pattern is invalid

        Performance: <1ms with cache, ~2-5ms cold
        """
        # Check cache first
        if self._cache:
            for pattern in self._cache.values():
                if pattern.name == pattern_name:
                    return pattern

        # Load from registry
        registry = self._load_registry()
        pattern_entry = None

        for entry in registry.get("patterns", []):
            if entry["name"] == pattern_name:
                pattern_entry = entry
                break

        if not pattern_entry:
            raise PatternNotFoundError(f"Pattern not found: {pattern_name}")

        # Load pattern file
        pattern = self.load_pattern_by_file(pattern_entry["file"])

        # Cache it
        self._cache[pattern.pattern_id] = pattern

        return pattern

    def load_pattern_by_file(self, relative_path: str) -> ModelPatternMetadata:
        """
        Load a pattern from a specific file path.

        Args:
            relative_path: Path relative to patterns_dir (e.g., "resilience/error_handling.yaml")

        Returns:
            ModelPatternMetadata object

        Raises:
            PatternNotFoundError: If file not found
            PatternValidationError: If parsing fails
        """
        file_path = self.patterns_dir / relative_path

        if not file_path.exists():
            raise PatternNotFoundError(f"Pattern file not found: {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data:
                raise PatternValidationError(f"Empty pattern file: {file_path}")

            # Convert YAML data to ModelPatternMetadata
            pattern = self._yaml_to_model(data, file_path)
            return pattern

        except yaml.YAMLError as e:
            raise PatternValidationError(f"Invalid YAML in {file_path}: {e}") from e
        except Exception as e:
            raise PatternLoaderError(
                f"Failed to load pattern from {file_path}: {e}"
            ) from e

    def get_patterns_by_category(
        self, category: EnumPatternCategory
    ) -> list[ModelPatternMetadata]:
        """
        Get all patterns in a specific category.

        Args:
            category: Pattern category

        Returns:
            List of patterns in the category

        Performance: <1ms with cache
        """
        patterns = self.load_all_patterns()
        return [p for p in patterns.values() if p.category == category]

    def get_patterns_by_node_type(
        self, node_type: EnumNodeType
    ) -> list[ModelPatternMetadata]:
        """
        Get all patterns applicable to a node type.

        Args:
            node_type: Node type (effect/compute/reducer/orchestrator)

        Returns:
            List of applicable patterns

        Performance: <1ms with cache
        """
        patterns = self.load_all_patterns()
        return [p for p in patterns.values() if node_type in p.applicable_to]

    def clear_cache(self) -> None:
        """Clear the pattern cache (useful for testing)."""
        self._cache.clear()
        self._registry_cache = None
        logger.debug("Pattern cache cleared")

    def _load_registry(self) -> dict:
        """Load the pattern registry YAML file."""
        if self._registry_cache:
            return self._registry_cache

        registry_path = self.patterns_dir / "registry.yaml"

        if not registry_path.exists():
            raise PatternLoaderError(f"Registry not found: {registry_path}")

        try:
            with open(registry_path, encoding="utf-8") as f:
                registry = yaml.safe_load(f)

            self._registry_cache = registry
            return registry

        except Exception as e:
            raise PatternLoaderError(
                f"Failed to load registry from {registry_path}: {e}"
            ) from e

    def _yaml_to_model(self, data: dict, file_path: Path) -> ModelPatternMetadata:
        """
        Convert YAML data to ModelPatternMetadata.

        Args:
            data: Parsed YAML data
            file_path: Source file path (for error messages)

        Returns:
            ModelPatternMetadata object

        Raises:
            PatternValidationError: If conversion fails
        """
        try:
            # Extract required fields
            name = data.get("name")
            if not name:
                raise PatternValidationError("Missing 'name' field")

            # Generate pattern_id (name + version)
            version = data.get("version", "1.0.0")
            pattern_id = f"{name}_v1"  # Simplified for now

            # Parse category
            category_str = data.get("category")
            try:
                category = EnumPatternCategory(category_str)
            except ValueError:
                raise PatternValidationError(f"Invalid category: {category_str}")

            # Parse applicable_to
            applicable_to_list = data.get("applicable_to", [])
            applicable_to = []
            for node_type_str in applicable_to_list:
                try:
                    applicable_to.append(EnumNodeType(node_type_str))
                except ValueError:
                    logger.warning(
                        f"Invalid node type '{node_type_str}' in {file_path}, skipping"
                    )

            if not applicable_to:
                raise PatternValidationError("No valid node types in 'applicable_to'")

            # Parse examples (optional)
            examples = []
            for example_data in data.get("examples", []):
                if isinstance(example_data, dict):
                    # Full example with metadata
                    examples.append(ModelPatternExample(**example_data))
                elif isinstance(example_data, str):
                    # Just node name - create minimal example
                    examples.append(
                        ModelPatternExample(
                            node_name=example_data,
                            node_type=applicable_to[0],  # Use first applicable type
                            code_snippet="# See pattern code_template",
                            description=f"Used in {example_data}",
                        )
                    )

            # Create ModelPatternMetadata
            pattern = ModelPatternMetadata(
                pattern_id=pattern_id,
                name=name,
                version=version,
                category=category,
                applicable_to=applicable_to,
                description=data.get("description", ""),
                code_template=data.get("code_template", ""),
                prerequisites=data.get("prerequisites", []),
                configuration=data.get("configuration", {}),
                examples=examples,
                tags=data.get("tags", []),
                complexity=data.get("complexity", 2),
                performance_impact=data.get("metrics", {}),
                use_cases=data.get("use_cases", []),
            )

            return pattern

        except PatternValidationError:
            raise
        except Exception as e:
            raise PatternValidationError(f"Failed to convert YAML to model: {e}") from e
