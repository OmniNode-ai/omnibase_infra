#!/usr/bin/env python3
"""Universal file stamping and OnexTree ingestion tool.

Supports multiple modes:
1. Pre-push hook: Stamp changed files before git push
2. Manual stamping: Stamp specific files, directories, or recursive paths
3. Backfill mode: Stamp entire repository for historical data

Usage:
    # Pre-push hook mode (called by git)
    ./scripts/stamp_and_ingest.py

    # Stamp specific file
    ./scripts/stamp_and_ingest.py --file src/module.py

    # Stamp directory recursively
    ./scripts/stamp_and_ingest.py --directory src/ --recursive

    # Backfill entire repository
    ./scripts/stamp_and_ingest.py --backfill

    # Stamp docs only
    ./scripts/stamp_and_ingest.py --directory docs/ --recursive --pattern "*.md"
"""

import argparse
import asyncio
import mimetypes
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)


# File type filters
BINARY_EXTENSIONS = {
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".svg",
    # Archives
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    # Executables
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    # Media
    ".mp4",
    ".avi",
    ".mov",
    ".mp3",
    ".wav",
    ".flac",
    # Databases
    ".db",
    ".sqlite",
    ".sqlite3",
    # Other
    ".pdf",
    ".pyc",
    ".pyo",
    ".whl",
}

TEXT_FILE_EXTENSIONS = {
    # Source code
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".go",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    # Configuration
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    # Documentation
    ".md",
    ".rst",
    ".txt",
    ".adoc",
    # Web
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".xml",
    # Shell
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    # Other
    ".sql",
    ".graphql",
    ".proto",
    ".thrift",
}


class StampingService:
    """Universal file stamping and ingestion service.

    Features:
    - Stamp any file type (text-based files)
    - Flexible file selection (file, directory, recursive, git-changed)
    - Backfill entire repository
    - Pattern matching for selective stamping
    - Graceful degradation on service unavailability
    """

    def __init__(
        self,
        stamping_url: str = "http://localhost:8053",
        onextree_url: str = "http://localhost:8054",
        skip_stamping: bool = False,
        skip_onextree: bool = False,
        namespace: str = "omninode.bridge.content",
    ):
        """Initialize stamping service.

        Args:
            stamping_url: MetadataStamping service URL
            onextree_url: OnexTree service URL
            skip_stamping: Skip stamping if service unavailable
            skip_onextree: Skip OnexTree ingestion if service unavailable
            namespace: Namespace for stamps
        """
        self.stamping_url = stamping_url
        self.onextree_url = onextree_url
        self.skip_stamping = skip_stamping
        self.skip_onextree = skip_onextree
        self.namespace = namespace
        self.correlation_id = uuid4()

        # Lazy-loaded clients
        self._stamping_client = None
        self._onextree_client = None

        # Statistics
        self.stats = {
            "total_files": 0,
            "stamped": 0,
            "skipped": 0,
            "failed": 0,
            "binary_skipped": 0,
            "already_stamped": 0,
        }

    async def _get_stamping_client(self):
        """Get or create stamping client."""
        if self._stamping_client is None:
            try:
                from omninode_bridge.clients.metadata_stamping_client import (
                    AsyncMetadataStampingClient,
                )

                self._stamping_client = AsyncMetadataStampingClient(
                    base_url=self.stamping_url,
                    timeout=10.0,
                    max_retries=2,
                )

                # Validate connection
                await self._stamping_client.connect()
                logger.info("MetadataStamping service connected", url=self.stamping_url)

            except Exception as e:
                logger.warning(
                    "MetadataStamping service unavailable, skipping stamping",
                    error=str(e),
                    url=self.stamping_url,
                )
                self.skip_stamping = True

        return self._stamping_client

    async def _get_onextree_client(self):
        """Get or create OnexTree client."""
        if self._onextree_client is None:
            try:
                from omninode_bridge.clients.onextree_client import AsyncOnexTreeClient

                self._onextree_client = AsyncOnexTreeClient(
                    base_url=self.onextree_url,
                    timeout=10.0,
                    max_retries=2,
                    enable_cache=True,
                )

                # Validate connection
                await self._onextree_client.connect()
                logger.info("OnexTree service connected", url=self.onextree_url)

            except Exception as e:
                logger.warning(
                    "OnexTree service unavailable, skipping ingestion",
                    error=str(e),
                    url=self.onextree_url,
                )
                self.skip_onextree = True

        return self._onextree_client

    def is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary.

        Args:
            file_path: Path to file

        Returns:
            True if file is binary
        """
        # Check extension first
        if file_path.suffix.lower() in BINARY_EXTENSIONS:
            return True

        # Check mimetype
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and not mime_type.startswith("text/"):
            return True

        # Try reading first 1KB to detect binary content
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                # Check for null bytes (common in binary files)
                if b"\x00" in chunk:
                    return True
        except Exception:
            pass

        return False

    def should_stamp_file(self, file_path: Path) -> bool:
        """Check if file should be stamped.

        Args:
            file_path: Path to file

        Returns:
            True if file should be stamped
        """
        # Skip if doesn't exist
        if not file_path.exists():
            return False

        # Skip directories
        if file_path.is_dir():
            return False

        # Skip hidden files
        if file_path.name.startswith("."):
            return False

        # Skip common directories
        skip_dirs = {
            ".git",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            "venv",
            ".venv",
        }
        if any(part in skip_dirs for part in file_path.parts):
            return False

        # Skip binary files
        return not self.is_binary_file(file_path)

    def get_changed_files_from_git(
        self, remote: str = "origin", branch: str = "main"
    ) -> list[Path]:
        """Get list of changed files since last push.

        Args:
            remote: Git remote name
            branch: Git branch name

        Returns:
            List of changed file paths
        """
        import subprocess

        try:
            # Get files changed since last push to remote
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{remote}/{branch}...HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )

            files = [Path(f.strip()) for f in result.stdout.splitlines() if f.strip()]

            # Filter for files that should be stamped
            stampable_files = [f for f in files if self.should_stamp_file(f)]

            logger.info(
                "Found changed files from git",
                total=len(files),
                stampable=len(stampable_files),
            )

            return stampable_files

        except subprocess.CalledProcessError as e:
            logger.error("Failed to get changed files from git", error=str(e))
            return []

    def get_files_from_path(
        self,
        path: Path,
        recursive: bool = False,
        pattern: Optional[str] = None,
    ) -> list[Path]:
        """Get files from a path (file or directory).

        Args:
            path: Path to file or directory
            recursive: If True, scan directory recursively
            pattern: Optional glob pattern (e.g., "*.md", "*.py")

        Returns:
            List of file paths
        """
        files = []

        if path.is_file():
            if self.should_stamp_file(path):
                files.append(path)

        elif path.is_dir():
            if recursive:
                # Recursive scan
                glob_pattern = pattern or "**/*"
                for file_path in path.glob(glob_pattern):
                    if self.should_stamp_file(file_path):
                        files.append(file_path)
            else:
                # Non-recursive scan
                glob_pattern = pattern or "*"
                for file_path in path.glob(glob_pattern):
                    if self.should_stamp_file(file_path):
                        files.append(file_path)

        logger.info(
            "Collected files from path",
            path=str(path),
            recursive=recursive,
            pattern=pattern,
            count=len(files),
        )

        return files

    def get_files_for_backfill(self, root_dir: Path = None) -> list[Path]:
        """Get all stampable files in repository for backfill.

        Args:
            root_dir: Root directory (defaults to current directory)

        Returns:
            List of all stampable file paths
        """
        if root_dir is None:
            root_dir = Path.cwd()

        logger.info("Starting backfill scan", root=str(root_dir))

        files = []
        for file_path in root_dir.glob("**/*"):
            if self.should_stamp_file(file_path):
                files.append(file_path)

        logger.info("Backfill scan complete", total_files=len(files))

        return files

    async def stamp_file(self, file_path: Path) -> dict[str, Any]:
        """Generate stamp for a file.

        Args:
            file_path: Path to file to stamp

        Returns:
            Stamp result dictionary
        """
        if self.skip_stamping:
            self.stats["skipped"] += 1
            return {"status": "skipped", "reason": "stamping_unavailable"}

        try:
            # Read file content
            content = file_path.read_bytes()

            # Get stamping client
            client = await self._get_stamping_client()
            if not client:
                self.stats["skipped"] += 1
                return {"status": "skipped", "reason": "client_unavailable"}

            # Generate hash
            hash_result = await client.generate_hash(
                file_data=content,
                namespace=self.namespace,
                correlation_id=self.correlation_id,
            )

            file_hash = hash_result["hash"]

            # Check if stamp already exists
            existing_stamp = await client.get_stamp(
                file_hash=file_hash,
                namespace=self.namespace,
                correlation_id=self.correlation_id,
            )

            if existing_stamp:
                self.stats["already_stamped"] += 1
                return {
                    "status": "existing",
                    "file_hash": file_hash,
                    "stamp": existing_stamp,
                }

            # Create new stamp
            stamp_data = {
                "file_path": str(file_path),
                "file_type": file_path.suffix,
                "file_name": file_path.name,
                "namespace": self.namespace,
                "source": "stamp_and_ingest_tool",
                "mime_type": mimetypes.guess_type(str(file_path))[0],
            }

            stamp_result = await client.create_stamp(
                file_hash=file_hash,
                file_path=str(file_path),
                file_size=len(content),
                stamp_data=stamp_data,
                namespace=self.namespace,
                correlation_id=self.correlation_id,
            )

            self.stats["stamped"] += 1

            return {
                "status": "created",
                "file_hash": file_hash,
                "stamp": stamp_result,
            }

        except Exception as e:
            logger.error("Failed to stamp file", file=str(file_path), error=str(e))
            self.stats["failed"] += 1
            return {"status": "failed", "error": str(e)}

    async def ingest_to_onextree(
        self, file_path: Path, stamp_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Ingest file patterns to OnexTree.

        Args:
            file_path: Path to file
            stamp_result: Stamp result from stamping service

        Returns:
            Ingestion result dictionary
        """
        if self.skip_onextree:
            return {"status": "skipped", "reason": "onextree_unavailable"}

        # Only ingest text files for now
        if file_path.suffix.lower() not in TEXT_FILE_EXTENSIONS:
            return {"status": "skipped", "reason": "non_text_file"}

        try:
            # Get OnexTree client
            client = await self._get_onextree_client()
            if not client:
                return {"status": "skipped", "reason": "client_unavailable"}

            # Read file content for analysis
            content = file_path.read_text()

            # Query OnexTree for intelligence on this file
            intelligence_result = await client.get_intelligence(
                context=f"File: {file_path}\nType: {file_path.suffix}\nContent preview: {content[:500]}",
                correlation_id=self.correlation_id,
            )

            return {
                "status": "ingested",
                "intelligence": intelligence_result,
            }

        except Exception as e:
            logger.error(
                "Failed to ingest to OnexTree", file=str(file_path), error=str(e)
            )
            return {"status": "failed", "error": str(e)}

    async def process_files(self, files: list[Path]) -> dict[str, Any]:
        """Process all files.

        Args:
            files: List of files to process

        Returns:
            Processing results summary
        """
        self.stats["total_files"] = len(files)
        details = []

        for idx, file_path in enumerate(files, 1):
            try:
                # Progress indicator
                if idx % 10 == 0 or idx == len(files):
                    print(f"Processing {idx}/{len(files)} files...", end="\r")

                # Stamp file
                stamp_result = await self.stamp_file(file_path)

                file_result = {
                    "file": str(file_path),
                    "stamp": stamp_result,
                }

                # Ingest to OnexTree
                ingest_result = await self.ingest_to_onextree(file_path, stamp_result)
                file_result["ingest"] = ingest_result

                details.append(file_result)

            except Exception as e:
                logger.error("Error processing file", file=str(file_path), error=str(e))
                self.stats["failed"] += 1
                details.append(
                    {
                        "file": str(file_path),
                        "error": str(e),
                    }
                )

        return {
            **self.stats,
            "details": details,
        }

    async def cleanup(self):
        """Cleanup resources."""
        if self._stamping_client:
            try:
                await self._stamping_client.disconnect()
            except Exception as e:
                logger.warning("Error disconnecting stamping client", error=str(e))

        if self._onextree_client:
            try:
                await self._onextree_client.disconnect()
            except Exception as e:
                logger.warning("Error disconnecting OnexTree client", error=str(e))


def print_summary(results: dict[str, Any]):
    """Print processing summary.

    Args:
        results: Processing results dictionary
    """
    print("\n" + "=" * 60)
    print("File Stamping & Ingestion Summary")
    print("=" * 60)
    print(f"Total files processed:    {results['total_files']}")
    print(f"Successfully stamped:     {results['stamped']}")
    print(f"Already stamped:          {results['already_stamped']}")
    print(f"Skipped:                  {results['skipped']}")
    print(f"Failed:                   {results['failed']}")
    print("=" * 60 + "\n")


async def main():
    """Main entry point."""
    import os

    parser = argparse.ArgumentParser(
        description="Universal file stamping and OnexTree ingestion tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-push mode (stamp changed files)
  %(prog)s

  # Stamp specific file
  %(prog)s --file src/module.py

  # Stamp directory recursively
  %(prog)s --directory src/ --recursive

  # Backfill entire repository
  %(prog)s --backfill

  # Stamp docs only
  %(prog)s --directory docs/ --recursive --pattern "*.md"
        """,
    )

    parser.add_argument(
        "--file",
        type=Path,
        help="Stamp a specific file",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        help="Stamp files in a directory",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan directory recursively",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Glob pattern for file selection (e.g., '*.md', '*.py')",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill entire repository",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="omninode.bridge.content",
        help="Namespace for stamps (default: omninode.bridge.content)",
    )
    parser.add_argument(
        "--skip-stamping",
        action="store_true",
        help="Skip stamping (for testing)",
    )
    parser.add_argument(
        "--skip-onextree",
        action="store_true",
        help="Skip OnexTree ingestion (for testing)",
    )

    args = parser.parse_args()

    # Get environment variables
    skip_stamping = (
        os.getenv("SKIP_STAMPING", "false").lower() == "true" or args.skip_stamping
    )
    skip_onextree = (
        os.getenv("SKIP_ONEXTREE", "false").lower() == "true" or args.skip_onextree
    )
    stamping_url = os.getenv("STAMPING_SERVICE_URL", "http://localhost:8053")
    onextree_url = os.getenv("ONEXTREE_SERVICE_URL", "http://localhost:8054")

    # Create service
    service = StampingService(
        stamping_url=stamping_url,
        onextree_url=onextree_url,
        skip_stamping=skip_stamping,
        skip_onextree=skip_onextree,
        namespace=args.namespace,
    )

    try:
        # Determine file selection mode
        files = []

        if args.file:
            # Single file mode
            if args.file.exists():
                files = service.get_files_from_path(args.file)
            else:
                print(f"Error: File not found: {args.file}")
                return 1

        elif args.directory:
            # Directory mode
            if args.directory.exists():
                files = service.get_files_from_path(
                    args.directory,
                    recursive=args.recursive,
                    pattern=args.pattern,
                )
            else:
                print(f"Error: Directory not found: {args.directory}")
                return 1

        elif args.backfill:
            # Backfill mode
            files = service.get_files_for_backfill()

        else:
            # Default: Pre-push mode (git changed files)
            files = service.get_changed_files_from_git()

        if not files:
            logger.info("No files to process")
            return 0

        print(f"Processing {len(files)} files...")

        # Process files
        results = await service.process_files(files)

        # Print summary
        print_summary(results)

        # Non-blocking for now - always allow push
        return 0

    except Exception as e:
        logger.error("Stamping failed", error=str(e))
        print(f"Error: {e}")
        return 0  # Non-blocking

    finally:
        await service.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
