"""File writing utilities for generated node scaffolding."""

from pathlib import Path
from typing import Dict, List

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError


class FileWriter:
    """Write generated node files to filesystem."""

    def __init__(self, output_base_dir: Path):
        """Initialize file writer.

        Args:
            output_base_dir: Base directory for generated files
        """
        self.output_base_dir = Path(output_base_dir)

    def write_files(
        self,
        file_contents: Dict[str, str],
        dry_run: bool = False,
    ) -> List[Path]:
        """Write generated files to filesystem.

        Args:
            file_contents: Dictionary mapping relative paths to content
            dry_run: If True, only log what would be written without writing

        Returns:
            List of paths that were (or would be) written

        Raises:
            OnexError: If file writing fails
        """
        written_paths = []

        for relative_path, content in file_contents.items():
            full_path = self.output_base_dir / relative_path

            if dry_run:
                print(f"[DRY RUN] Would write: {full_path}")
                written_paths.append(full_path)
                continue

            try:
                # Create parent directories
                full_path.parent.mkdir(parents=True, exist_ok=True)

                # Write file
                full_path.write_text(content, encoding="utf-8")

                written_paths.append(full_path)

            except Exception as e:
                raise OnexError(
                    code=CoreErrorCode.PROCESSING_ERROR,
                    message=f"Failed to write file: {full_path}",
                    details={"error": str(e)},
                ) from e

        return written_paths

    def create_init_files(
        self,
        directories: List[Path],
        dry_run: bool = False,
    ) -> List[Path]:
        """Create __init__.py files in specified directories.

        Args:
            directories: List of directory paths
            dry_run: If True, only log without writing

        Returns:
            List of __init__.py paths created

        Raises:
            OnexError: If init file creation fails
        """
        init_paths = []

        for directory in directories:
            init_path = directory / "__init__.py"

            if dry_run:
                print(f"[DRY RUN] Would create: {init_path}")
                init_paths.append(init_path)
                continue

            try:
                directory.mkdir(parents=True, exist_ok=True)

                if not init_path.exists():
                    init_path.write_text(
                        '"""Package initialization."""\n',
                        encoding="utf-8"
                    )

                init_paths.append(init_path)

            except Exception as e:
                raise OnexError(
                    code=CoreErrorCode.PROCESSING_ERROR,
                    message=f"Failed to create __init__.py: {init_path}",
                    details={"error": str(e)},
                ) from e

        return init_paths
