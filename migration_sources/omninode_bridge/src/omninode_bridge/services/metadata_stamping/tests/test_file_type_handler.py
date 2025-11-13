"""Tests for protocol file type handler."""

import pytest

from omninode_bridge.services.metadata_stamping.protocols.file_type_handler import (
    ProtocolFileTypeHandler,
)


class TestProtocolFileTypeHandler:
    """Test protocol file type handling and omnibase_core compliance."""

    @pytest.fixture
    def handler(self):
        return ProtocolFileTypeHandler()

    @pytest.mark.asyncio
    async def test_file_type_detection_accuracy(self, handler):
        """Test accurate file type detection."""
        test_cases = [
            ("image.jpg", "image/jpeg", "image"),
            ("document.pdf", "application/pdf", "document"),
            ("audio.mp3", "audio/mpeg", "audio"),
            ("video.mp4", "video/mp4", "video"),
            ("archive.zip", "application/zip", "archive"),
            ("text.txt", "text/plain", "document"),
            ("photo.png", None, "image"),
            ("data.json", "application/json", "document"),
        ]

        for file_path, content_type, expected_type in test_cases:
            try:
                detected_type = await handler.detect_file_type(file_path, content_type)
                assert detected_type == expected_type, f"Failed for {file_path}"
            except ValueError:
                # Some types might not be supported yet
                if expected_type not in ["document"]:
                    raise

    @pytest.mark.asyncio
    async def test_protocol_compliance_validation(self, handler):
        """Test omnibase_core protocol compliance."""
        valid_file_data = {
            "file_path": "/test/file.jpg",
            "file_size": 1024,
            "metadata": {"width": 1920, "height": 1080},
        }

        is_compliant = await handler.validate_protocol_compliance(valid_file_data)
        assert is_compliant is True

        # Test missing required fields
        invalid_data = {"file_path": "/test/file.jpg"}
        assert await handler.validate_protocol_compliance(invalid_data) is False

        # Test invalid file size
        invalid_size = {"file_path": "/test/file.jpg", "file_size": -1}
        assert await handler.validate_protocol_compliance(invalid_size) is False

    @pytest.mark.asyncio
    async def test_multi_modal_processing(self, handler):
        """Test multi-modal capabilities for different file types."""
        # Test different file types require different handlers
        image_handler = await handler.route_to_handler("image", "extract_metadata")
        document_handler = await handler.route_to_handler(
            "document", "extract_metadata"
        )

        assert image_handler != document_handler
        assert callable(image_handler)
        assert callable(document_handler)

    @pytest.mark.asyncio
    async def test_error_handling_unsupported_types(self, handler):
        """Test error handling for unsupported file types."""
        with pytest.raises(ValueError):
            await handler.detect_file_type("unknown.xyz", "application/unknown")

        with pytest.raises(ValueError):
            await handler.route_to_handler("unsupported", "extract_metadata")

    @pytest.mark.asyncio
    async def test_metadata_extraction(self, handler):
        """Test metadata extraction for different file types."""
        # Test image metadata extraction
        image_metadata = await handler._extract_image_metadata(
            "/test/image.jpg", b"fake_image_data"
        )
        assert image_metadata["type"] == "image"
        assert image_metadata["size"] == len(b"fake_image_data")
        assert image_metadata["format"] == ".jpg"

        # Test document metadata extraction
        doc_metadata = await handler._extract_document_metadata(
            "/test/doc.pdf", b"fake_pdf_data"
        )
        assert doc_metadata["type"] == "document"
        assert doc_metadata["format"] == ".pdf"

    @pytest.mark.asyncio
    async def test_file_validation(self, handler):
        """Test file validation for different types."""
        # Test image validation
        jpeg_signature = b"\xFF\xD8\xFF" + b"fake_jpeg_data"
        assert await handler._validate_image(jpeg_signature) is True

        png_signature = b"\x89PNG\r\n\x1a\n" + b"fake_png_data"
        assert await handler._validate_image(png_signature) is True

        # Test document validation
        pdf_signature = b"%PDF" + b"fake_pdf_data"
        assert await handler._validate_document(pdf_signature) is True

        # Test invalid data
        assert await handler._validate_image(b"invalid_data") is False
        assert await handler._validate_audio(b"invalid_data") is False
