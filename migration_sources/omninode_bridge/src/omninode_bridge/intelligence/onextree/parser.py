"""
YAML parser and validator for .onextree files.

Handles reading and writing .onextree files with Pydantic validation.
"""

from pathlib import Path
from typing import Optional

import structlog
import yaml

from .models import ModelOnextreeRoot

logger = structlog.get_logger(__name__)


class ToolOnextreeProcessor:
    """
    Parser and validator for .onextree YAML files.

    Provides conversion between YAML files and Pydantic models with
    validation and error handling.
    """

    @staticmethod
    def parse_onextree_file(file_path: Path) -> Optional[ModelOnextreeRoot]:
        """
        Parse .onextree YAML file into Pydantic model.

        Args:
            file_path: Path to .onextree file

        Returns:
            Parsed ModelOnextreeRoot or None if invalid

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is malformed
            ValueError: If Pydantic validation fails
        """
        try:
            if not file_path.exists():
                logger.error("File not found", file_path=str(file_path))
                return None

            with open(file_path) as f:
                raw_data = yaml.safe_load(f)

            if not raw_data:
                logger.error("Empty or invalid YAML file", file_path=str(file_path))
                return None

            # Validate with Pydantic
            tree_root = ModelOnextreeRoot.model_validate(raw_data)
            return tree_root

        except yaml.YAMLError as e:
            logger.error("YAML parsing error", file_path=str(file_path), error=str(e))
            return None
        except ValueError as e:
            logger.error("Validation error", file_path=str(file_path), error=str(e))
            return None
        except Exception as e:
            logger.error(
                "Failed to parse .onextree file",
                file_path=str(file_path),
                error=str(e),
                exc_info=True,
            )
            return None

    @staticmethod
    def write_onextree_file(tree_root: ModelOnextreeRoot, file_path: Path) -> bool:
        """
        Write ModelOnextreeRoot to .onextree YAML file.

        Args:
            tree_root: Tree root model to serialize
            file_path: Output file path

        Returns:
            True if successful, False otherwise

        Raises:
            IOError: If file cannot be written
        """
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict with Pydantic
            # Use mode='json' to ensure datetime serialization
            data = tree_root.model_dump(mode="json", exclude_none=True)

            # Write YAML with pretty formatting
            with open(file_path, "w") as f:
                yaml.safe_dump(
                    data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    indent=2,
                )

            return True

        except OSError as e:
            logger.error(
                "IO error writing .onextree file",
                file_path=str(file_path),
                error=str(e),
            )
            return False
        except Exception as e:
            logger.error(
                "Failed to write .onextree file",
                file_path=str(file_path),
                error=str(e),
                exc_info=True,
            )
            return False

    @staticmethod
    def validate_onextree_schema(data: dict) -> bool:
        """
        Validate data against OnexTree schema without full parsing.

        Args:
            data: Dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            ModelOnextreeRoot.model_validate(data)
            return True
        except ValueError:
            return False
