# === OmniNode:Tool_Metadata ===
# metadata_version: 0.1
# name: metadata_stamping_engine
# title: MetadataStampingService Core Engine
# version: 0.1.0
# namespace: omninode.services.metadata
# category: service.infrastructure.stamping
# kind: service
# role: stamping_engine
# description: |
#   Core stamping engine implementing multi-modal stamping capabilities with
#   BLAKE3 hashing, lightweight and rich metadata stamp generation, and
#   stamp validation functionality for high-performance operations.
# tags: [engine, metadata, blake3, stamping, validation, hashing]
# author: OmniNode Development Team
# license: MIT
# entrypoint: stamping_engine.py
# protocols_supported: [O.N.E. v0.1]
# runtime_constraints: {sandboxed: false, privileged: false, requires_network: false, requires_gpu: false}
# dependencies: [{"name": "blake3", "version": "^0.4.1"}]
# environment: [python>=3.11]
# === /OmniNode:Tool_Metadata ===

"""Core stamping engine for metadata operations.

This module implements the multi-modal stamping engine supporting both
lightweight and rich metadata stamping capabilities.
"""

import logging
import re
import uuid
from datetime import UTC, datetime
from typing import Any, Optional

from ..protocols.file_type_handler import ProtocolFileTypeHandler
from .hash_generator import BLAKE3HashGenerator

logger = logging.getLogger(__name__)


class StampingEngine:
    """Core stamping engine supporting multiple stamping modes."""

    def __init__(self):
        """Initialize the stamping engine."""
        self.hash_generator = BLAKE3HashGenerator()
        self.file_handler = ProtocolFileTypeHandler()

        # Stamp patterns for extraction
        self.lightweight_pattern = re.compile(
            r"# ONEX:uid=([a-zA-Z0-9_-]+),hash=([a-zA-Z0-9]+),ts=([0-9T:\-\+\.]+).*"
        )
        self.rich_pattern = re.compile(
            r"<!--\s*ONEX_METADATA_START\s*-->(.*?)<!--\s*ONEX_METADATA_END\s*-->",
            re.DOTALL,
        )

    async def create_lightweight_stamp(
        self, content_hash: str, metadata: dict[str, Any]
    ) -> str:
        """Create lightweight stamp for performance-critical scenarios.

        Format: # ONEX:uid=uuid,hash=blake3,ts=timestamp

        Args:
            content_hash: BLAKE3 hash of the content
            metadata: Additional metadata to include

        Returns:
            Lightweight stamp string
        """
        stamp_uid = metadata.get("uid", str(uuid.uuid4()))
        timestamp = metadata.get("timestamp", datetime.now(UTC).isoformat())

        stamp = f"# ONEX:uid={stamp_uid},hash={content_hash},ts={timestamp}"

        # Add optional metadata if present
        if "author" in metadata:
            stamp += f",author={metadata['author']}"
        if "version" in metadata:
            stamp += f",v={metadata['version']}"

        return stamp

    async def create_rich_stamp(
        self, content_hash: str, metadata: dict[str, Any]
    ) -> str:
        """Create rich metadata block with full YAML metadata.

        Args:
            content_hash: BLAKE3 hash of the content
            metadata: Comprehensive metadata dictionary

        Returns:
            Rich metadata stamp block
        """
        stamp_uid = metadata.get("uid", str(uuid.uuid4()))
        timestamp = metadata.get("timestamp", datetime.now(UTC).isoformat())

        # Build rich metadata block
        stamp_lines = [
            "<!-- ONEX_METADATA_START -->",
            "<!-- ONEX Metadata Stamp",
            f"  uid: {stamp_uid}",
            f"  hash: {content_hash}",
            f"  timestamp: {timestamp}",
            f"  protocol_version: {metadata.get('protocol_version', '1.0')}",
        ]

        # Add file information if present
        if "file_path" in metadata:
            stamp_lines.append(f"  file_path: {metadata['file_path']}")
        if "file_size" in metadata:
            stamp_lines.append(f"  file_size: {metadata['file_size']}")
        if "content_type" in metadata:
            stamp_lines.append(f"  content_type: {metadata['content_type']}")

        # Add integrity information
        stamp_lines.extend(
            [
                "  integrity:",
                "    algorithm: BLAKE3",
                f"    hash: {content_hash}",
            ]
        )

        # Close the metadata block
        stamp_lines.extend(["-->", "<!-- ONEX_METADATA_END -->"])

        return "\n".join(stamp_lines)

    async def stamp_content(
        self,
        content: str,
        file_path: Optional[str] = None,
        stamp_type: str = "hash_only",
        metadata: Optional[dict[str, Any]] = None,
        enable_langextract: bool = True,
    ) -> dict[str, Any]:
        """Stamp content with metadata using hash-only approach.

        This follows the omnibase_3 pattern:
        1. Extract enhanced metadata using LangExtract (if enabled)
        2. Store full metadata externally
        3. Embed only lightweight hash reference in content

        Args:
            content: Content to stamp
            file_path: File path (required for hash-only stamping)
            stamp_type: Type of stamp ("hash_only", "lightweight", or "rich")
            metadata: Additional metadata
            enable_langextract: Whether to use LangExtract for enhanced metadata

        Returns:
            Stamped content and stamp information with enhanced metadata
        """
        # Remove existing stamps for clean hashing
        clean_content = await self._remove_existing_stamps(content)

        # Generate content hash
        content_bytes = clean_content.encode("utf-8")
        hash_result = await self.hash_generator.generate_hash(content_bytes, file_path)
        content_hash = hash_result["hash"]

        # Prepare basic metadata
        stamp_metadata = metadata or {}
        stamp_metadata["file_path"] = file_path
        stamp_metadata["file_size"] = len(content_bytes)

        # Detect file type if file_path is provided
        if file_path:
            try:
                file_type = await self.file_handler.detect_file_type(file_path)
                stamp_metadata["file_type"] = file_type
            except ValueError:
                # File type detection failed, continue without it
                pass

        # Enhanced metadata extraction with LangExtract
        enhanced_metadata = None
        hash_reference = None

        if stamp_type == "hash_only" and enable_langextract and file_path:
            try:
                # Initialize enhanced metadata extractor if not exists
                if not hasattr(self, "enhanced_extractor"):
                    from .enhanced_metadata_extractor import EnhancedMetadataExtractor

                    self.enhanced_extractor = EnhancedMetadataExtractor(
                        enable_dot_files=True,
                        database_client=getattr(self, "db_client", None),
                    )

                # Extract and store enhanced metadata
                hash_reference, enhanced_metadata = (
                    await self.enhanced_extractor.extract_and_store_metadata(
                        content=clean_content,
                        file_path=file_path,
                        content_type=stamp_metadata.get("file_type"),
                    )
                )

                # Create lightweight stamp with hash reference
                stamp = hash_reference.to_lightweight_stamp()
                stamped_content = stamp + "\n" + content

                return {
                    "stamped_content": stamped_content,
                    "stamp": stamp,
                    "stamp_type": "hash_only",
                    "content_hash": content_hash,
                    "metadata_hash": hash_reference.metadata_hash,
                    "hash_reference": hash_reference.model_dump(),
                    "enhanced_metadata": (
                        enhanced_metadata.model_dump() if enhanced_metadata else None
                    ),
                    "categories": (
                        [cat.model_dump() for cat in enhanced_metadata.categories]
                        if enhanced_metadata
                        else []
                    ),
                    "tags": (
                        [tag.model_dump() for tag in enhanced_metadata.tags]
                        if enhanced_metadata
                        else []
                    ),
                    "entities": (
                        [entity.model_dump() for entity in enhanced_metadata.entities]
                        if enhanced_metadata
                        else []
                    ),
                    "processing_time_ms": (
                        enhanced_metadata.processing_time_ms
                        if enhanced_metadata
                        else hash_result.get("execution_time_ms")
                    ),
                    "performance_metrics": {
                        "hash_generation_ms": hash_result.get("execution_time_ms", 0),
                        "langextract_processing_ms": (
                            enhanced_metadata.processing_time_ms
                            if enhanced_metadata
                            else 0
                        ),
                        "total_processing_ms": (
                            hash_result.get("execution_time_ms", 0)
                            + (
                                enhanced_metadata.processing_time_ms
                                if enhanced_metadata
                                else 0
                            )
                        ),
                    },
                }

            except ImportError:
                logger.warning(
                    "LangExtract not available, falling back to lightweight stamping"
                )
                stamp_type = "lightweight"
            except Exception as e:
                logger.error(
                    f"Enhanced metadata extraction failed: {e}, falling back to lightweight stamping"
                )
                stamp_type = "lightweight"

        # Fallback to traditional stamping methods
        if stamp_type == "lightweight":
            stamp = await self.create_lightweight_stamp(content_hash, stamp_metadata)
        else:  # rich
            stamp = await self.create_rich_stamp(content_hash, stamp_metadata)

        # Apply stamp to content
        if stamp_type == "lightweight":
            # Add lightweight stamp at the beginning
            stamped_content = stamp + "\n" + content
        else:
            # Add rich stamp at the end
            stamped_content = content + "\n\n" + stamp

        return {
            "stamped_content": stamped_content,
            "stamp": stamp,
            "content_hash": content_hash,
            "stamp_type": stamp_type,
            "execution_time_ms": hash_result["execution_time_ms"],
            "performance_grade": hash_result["performance_grade"],
        }

    async def validate_stamp(
        self, content: str, expected_hash: Optional[str] = None
    ) -> dict[str, Any]:
        """Validate existing stamps in content.

        Args:
            content: Content with stamps to validate
            expected_hash: Optional expected hash for validation

        Returns:
            Validation results
        """
        # Extract existing stamps
        stamps = await self.extract_stamps(content)

        if not stamps:
            return {
                "valid": False,
                "reason": "No stamps found in content",
                "stamps_found": 0,
            }

        # Remove stamps and calculate current hash
        clean_content = await self._remove_existing_stamps(content)
        content_bytes = clean_content.encode("utf-8")
        hash_result = await self.hash_generator.generate_hash(content_bytes)
        current_hash = hash_result["hash"]

        # Validate stamps
        validation_results = []
        for stamp in stamps:
            stamp_hash = stamp.get("hash")
            is_valid = stamp_hash == current_hash

            # If expected_hash is provided, validate against it
            if expected_hash and stamp_hash != expected_hash:
                is_valid = False

            validation_results.append(
                {
                    "stamp_type": stamp["type"],
                    "stamp_hash": stamp_hash,
                    "is_valid": is_valid,
                    "current_hash": current_hash,
                }
            )

        # Overall validation
        all_valid = all(result["is_valid"] for result in validation_results)

        return {
            "valid": all_valid,
            "stamps_found": len(stamps),
            "current_hash": current_hash,
            "validation_results": validation_results,
            "execution_time_ms": hash_result["execution_time_ms"],
        }

    async def extract_stamps(self, content: str) -> list:
        """Extract all stamps from content.

        Args:
            content: Content to extract stamps from

        Returns:
            List of extracted stamp data
        """
        stamps = []

        # Extract lightweight stamps
        for match in self.lightweight_pattern.finditer(content):
            stamps.append(
                {
                    "type": "lightweight",
                    "uid": match.group(1),
                    "hash": match.group(2),
                    "timestamp": match.group(3),
                    "raw": match.group(0),
                }
            )

        # Extract rich stamps
        for match in self.rich_pattern.finditer(content):
            stamp_content = match.group(1)
            # Parse YAML-like content
            stamp_data = self._parse_rich_stamp(stamp_content)
            stamp_data["type"] = "rich"
            stamp_data["raw"] = match.group(0)
            stamps.append(stamp_data)

        return stamps

    async def _remove_existing_stamps(self, content: str) -> str:
        """Remove existing stamps from content for clean hashing.

        Args:
            content: Content with potential stamps

        Returns:
            Content with stamps removed
        """
        # Remove lightweight stamps
        clean_content = self.lightweight_pattern.sub("", content)

        # Remove rich stamps
        clean_content = self.rich_pattern.sub("", clean_content)

        # Clean up any extra blank lines left behind
        clean_content = re.sub(r"\n{3,}", "\n\n", clean_content)

        return clean_content.strip()

    def _parse_rich_stamp(self, stamp_content: str) -> dict[str, Any]:
        """Parse rich stamp content into dictionary.

        Args:
            stamp_content: Rich stamp content to parse

        Returns:
            Parsed stamp data
        """
        stamp_data = {}
        lines = stamp_content.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("<!--"):
                continue

            # Parse key-value pairs
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Store in stamp_data
                if key and value:
                    stamp_data[key] = value

        return stamp_data

    async def cleanup(self):
        """Clean up resources."""
        await self.hash_generator.cleanup()
