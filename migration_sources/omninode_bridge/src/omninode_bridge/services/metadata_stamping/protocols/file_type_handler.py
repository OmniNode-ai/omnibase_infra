"""Protocol File Type Handler for omnibase_core compliance.

This module implements the standardized protocol interface for file type handling
and validation, ensuring full compliance with omnibase_core requirements.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class FileTypeHandlerProtocol(Protocol):
    """Protocol interface for file type handlers ensuring omnibase_core compliance.

    This protocol defines the contract that all file type handlers must implement
    for proper integration with the metadata stamping service.
    """

    async def detect_file_type(
        self, file_path: str, content_type: Optional[str] = None
    ) -> str:
        """Detect file type based on extension and content analysis.

        Args:
            file_path: Path to the file
            content_type: Optional MIME content type

        Returns:
            Detected file type category

        Raises:
            ValueError: If file type cannot be determined
        """
        ...

    async def validate_protocol_compliance(self, file_data: dict) -> bool:
        """Validate file data against omnibase_core protocol requirements.

        Args:
            file_data: Dictionary containing file metadata

        Returns:
            True if compliant, False otherwise
        """
        ...

    async def route_to_handler(self, file_type: str, operation: str) -> Callable:
        """Route to appropriate handler based on file type and operation.

        Args:
            file_type: Type of file
            operation: Operation to perform

        Returns:
            Handler function for the operation

        Raises:
            ValueError: If handler not found
        """
        ...


class ProtocolFileTypeHandler:
    """Handles file type detection and protocol routing for omnibase_core compliance."""

    def __init__(self):
        """Initialize the protocol file type handler."""
        self.supported_types = {
            "document": [
                ".pdf",
                ".doc",
                ".docx",
                ".txt",
                ".md",
                ".py",
                ".js",
                ".json",
                ".yaml",
                ".yml",
            ],
            "archive": [".zip", ".tar", ".gz", ".rar", ".7z"],
        }

        # Handler registry for different file types and operations
        self._handlers = {}
        self._initialize_handlers()

    def _initialize_handlers(self):
        """Initialize handlers for different file types."""
        # Register default handlers
        self._handlers["document"] = {
            "extract_metadata": self._extract_document_metadata,
            "validate": self._validate_document,
            "process": self._process_document,
        }
        self._handlers["archive"] = {
            "extract_metadata": self._extract_archive_metadata,
            "validate": self._validate_archive,
            "process": self._process_archive,
        }

    async def detect_file_type(
        self, file_path: str, content_type: Optional[str] = None
    ) -> str:
        """Detect file type based on extension and content analysis.

        Args:
            file_path: Path to the file
            content_type: Optional MIME content type (str or bytes)

        Returns:
            Detected file type category

        Raises:
            ValueError: If file type cannot be determined
        """
        # Extract file extension
        path = Path(file_path)
        extension = path.suffix.lower()

        # Check against supported types
        for file_type, extensions in self.supported_types.items():
            if extension in extensions:
                return file_type

        # Fallback to content type analysis if provided
        if content_type:
            # Handle both string and bytes content_type
            if isinstance(content_type, bytes):
                content_type = content_type.decode("utf-8", errors="ignore")

            type_mapping = {
                "application/pdf": "document",
                "text/": "document",
                "application/zip": "archive",
                "application/x-": "archive",
            }

            for pattern, file_type in type_mapping.items():
                if content_type.startswith(pattern):
                    return file_type

        raise ValueError(
            f"Unsupported file type: {extension} (content_type: {content_type})"
        )

    async def validate_protocol_compliance(self, file_data: dict) -> bool:
        """Validate file data against omnibase_core protocol requirements.

        Args:
            file_data: Dictionary containing file metadata

        Returns:
            True if compliant, False otherwise
        """
        required_fields = ["file_path", "file_size"]

        # Check required fields
        for field in required_fields:
            if field not in file_data:
                return False

        # Validate file path
        if not file_data.get("file_path"):
            return False

        # Validate file size
        file_size = file_data.get("file_size")
        if not isinstance(file_size, int) or file_size < 0:
            return False

        # Validate metadata if present
        metadata = file_data.get("metadata")
        return not (metadata and not isinstance(metadata, dict))

    async def route_to_handler(self, file_type: str, operation: str) -> Callable:
        """Route to appropriate handler based on file type and operation.

        Args:
            file_type: Type of file
            operation: Operation to perform

        Returns:
            Handler function for the operation

        Raises:
            ValueError: If handler not found
        """
        if file_type not in self._handlers:
            raise ValueError(f"No handler registered for file type: {file_type}")

        type_handlers = self._handlers[file_type]
        if operation not in type_handlers:
            raise ValueError(
                f"No handler for operation '{operation}' on type '{file_type}'"
            )

        return type_handlers[operation]

    # Document and archive handler implementations
    async def _extract_document_metadata(
        self, file_path: str, content: bytes
    ) -> dict[str, Any]:
        """Extract metadata from document files.

        Args:
            file_path: Path to the document file
            content: File content

        Returns:
            Dictionary of extracted metadata
        """
        return {
            "type": "document",
            "size": len(content),
            "path": file_path,
            "format": Path(file_path).suffix.lower(),
            "encoding": "utf-8",  # Assumed
            "pages": None,  # Would extract for PDFs
        }

    async def _validate_document(self, content: bytes) -> bool:
        """Validate document file integrity.

        Args:
            content: Document content

        Returns:
            True if valid, False otherwise
        """
        # Check for PDF signature
        if content.startswith(b"%PDF"):
            return True

        # Check for text documents (basic validation)
        try:
            content.decode("utf-8")
            return True
        except UnicodeDecodeError:
            pass

        return False

    async def _process_document(self, content: bytes) -> bytes:
        """Process document content.

        Args:
            content: Document content

        Returns:
            Processed content
        """
        return content

    async def _extract_archive_metadata(
        self, file_path: str, content: bytes
    ) -> dict[str, Any]:
        """Extract metadata from archive files.

        Args:
            file_path: Path to the archive file
            content: File content

        Returns:
            Dictionary of extracted metadata
        """
        return {
            "type": "archive",
            "size": len(content),
            "path": file_path,
            "format": Path(file_path).suffix.lower(),
            "compression": None,
            "files_count": None,
            "uncompressed_size": None,
        }

    async def _validate_archive(self, content: bytes) -> bool:
        """Validate archive file integrity.

        Args:
            content: Archive content

        Returns:
            True if valid, False otherwise
        """
        # Check for archive signatures
        signatures = {
            b"PK\x03\x04": "zip",
            b"PK\x05\x06": "zip",
            b"Rar!": "rar",
            b"7z\xBC\xAF\x27\x1C": "7z",
        }

        for signature in signatures:
            if content.startswith(signature):
                return True

        # Check for tar/gzip
        return content.startswith(b"\x1F\x8B")  # gzip

    async def _process_archive(self, content: bytes) -> bytes:
        """Process archive content.

        Args:
            content: Archive content

        Returns:
            Processed content
        """
        return content
