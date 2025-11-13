# === OmniNode:Tool_Metadata ===
# metadata_version: 0.1
# name: enhanced_metadata_api_router
# title: Enhanced Metadata API Router with LangExtract
# version: 0.1.0
# namespace: omninode.services.metadata
# category: service.infrastructure.api
# kind: service
# role: enhanced_api_router
# description: |
#   FastAPI router for enhanced metadata capabilities with LangExtract integration,
#   hash-only stamping, category/tag extraction, and intelligent content analysis.
# tags: [api, router, langextract, categories, tags, intelligence, hash-only]
# author: OmniNode Development Team
# license: MIT
# protocols_supported: [O.N.E. v0.1]
# runtime_constraints: {sandboxed: false, privileged: false, requires_network: true, requires_gpu: false}
# dependencies: [{"name": "fastapi", "version": "^0.115.0"}, {"name": "langextract", "version": "^1.0.7"}]
# environment: [python>=3.12]
# === /OmniNode:Tool_Metadata ===

"""Enhanced metadata API endpoints with LangExtract integration.

This module provides advanced metadata extraction and querying capabilities
including intelligent categorization, tagging, entity extraction, and
hash-only stamping with external metadata storage.
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Query, UploadFile
from pydantic import BaseModel, Field

from ..engine.enhanced_metadata_extractor import EnhancedMetadataExtractor
from ..models.responses import UnifiedResponse

logger = logging.getLogger(__name__)

# Create enhanced router
enhanced_router = APIRouter(
    prefix="/api/v1/enhanced-metadata", tags=["enhanced-metadata"]
)


# Request/Response Models
class EnhancedStampRequest(BaseModel):
    """Request model for enhanced stamping with LangExtract."""

    content: str = Field(..., description="Content to stamp and analyze")
    file_path: str = Field(
        ..., description="File path for context and dot file creation"
    )
    content_type: Optional[str] = Field(None, description="Content type hint")
    enable_langextract: bool = Field(True, description="Enable LangExtract analysis")
    enable_dot_files: bool = Field(True, description="Create dot files with metadata")
    custom_extraction_prompt: Optional[str] = Field(
        None, description="Custom extraction prompt"
    )


class CategorySearchRequest(BaseModel):
    """Request model for searching by categories."""

    categories: list[str] = Field(..., description="Categories to search for")
    confidence_threshold: float = Field(0.7, description="Minimum confidence threshold")


class TagSearchRequest(BaseModel):
    """Request model for searching by tags."""

    tags: list[str] = Field(..., description="Tags to search for")
    tag_types: Optional[list[str]] = Field(None, description="Tag types to filter by")
    relevance_threshold: float = Field(0.7, description="Minimum relevance threshold")


class EnhancedStampResponse(BaseModel):
    """Response model for enhanced stamping."""

    # Stamp information
    stamped_content: str = Field(..., description="Content with lightweight stamp")
    lightweight_stamp: str = Field(..., description="The lightweight stamp applied")
    metadata_hash: str = Field(..., description="Hash of the enhanced metadata")
    content_hash: str = Field(..., description="Hash of the original content")

    # Enhanced metadata
    categories: list[dict[str, Any]] = Field(
        default_factory=list, description="Extracted categories"
    )
    tags: list[dict[str, Any]] = Field(
        default_factory=list, description="Extracted tags"
    )
    entities: list[dict[str, Any]] = Field(
        default_factory=list, description="Extracted entities"
    )

    # Analysis results
    content_type_detected: Optional[str] = Field(
        None, description="AI-detected content type"
    )
    language_detected: Optional[str] = Field(None, description="Detected language")
    complexity_score: Optional[float] = Field(
        None, description="Content complexity score"
    )

    # Processing information
    extraction_model: Optional[str] = Field(None, description="LLM model used")
    extraction_confidence: float = Field(
        0.0, description="Overall extraction confidence"
    )
    processing_time_ms: Optional[float] = Field(None, description="Processing time")

    # Storage information
    dot_file_created: Optional[str] = Field(
        None, description="Path to created dot file"
    )
    database_stored: bool = Field(False, description="Whether stored in database")


class MetadataQueryResponse(BaseModel):
    """Response model for metadata queries."""

    metadata_hash: str = Field(..., description="Metadata hash")
    content_hash: str = Field(..., description="Content hash")
    enhanced_metadata: dict[str, Any] = Field(..., description="Full enhanced metadata")
    created_at: str = Field(..., description="Creation timestamp")


# Service dependency
_enhanced_extractor: Optional[EnhancedMetadataExtractor] = None


async def get_enhanced_extractor() -> EnhancedMetadataExtractor:
    """Dependency to get enhanced metadata extractor."""
    global _enhanced_extractor
    if _enhanced_extractor is None:
        _enhanced_extractor = EnhancedMetadataExtractor(
            model_id="gemini-2.5-flash",
            enable_dot_files=True,
            enable_visualization=True,
        )
    return _enhanced_extractor


# Enhanced API Endpoints


@enhanced_router.post("/stamp-with-intelligence", response_model=UnifiedResponse)
async def stamp_with_intelligence(
    request: EnhancedStampRequest,
    extractor: EnhancedMetadataExtractor = Depends(get_enhanced_extractor),
) -> UnifiedResponse:
    """Stamp content with intelligent metadata extraction using LangExtract.

    This endpoint provides the enhanced stamping experience:
    1. Extracts categories, tags, and entities using LangExtract
    2. Stores full metadata externally (database + dot files)
    3. Returns content with only lightweight hash stamp
    4. Provides rich metadata in response for immediate use
    """
    try:
        # Extract enhanced metadata and create hash reference
        hash_ref, enhanced_metadata = await extractor.extract_and_store_metadata(
            content=request.content,
            file_path=request.file_path,
            content_type=request.content_type,
            custom_prompt=request.custom_extraction_prompt,
        )

        # Create stamped content with lightweight stamp
        stamped_content = hash_ref.to_lightweight_stamp() + "\n" + request.content

        # Prepare response data
        response_data = EnhancedStampResponse(
            stamped_content=stamped_content,
            lightweight_stamp=hash_ref.to_lightweight_stamp(),
            metadata_hash=hash_ref.metadata_hash,
            content_hash=hash_ref.content_hash,
            categories=[cat.model_dump() for cat in enhanced_metadata.categories],
            tags=[tag.model_dump() for tag in enhanced_metadata.tags],
            entities=[entity.model_dump() for entity in enhanced_metadata.entities],
            content_type_detected=enhanced_metadata.content_type_detected,
            language_detected=enhanced_metadata.language_detected,
            complexity_score=enhanced_metadata.complexity_score,
            extraction_model=enhanced_metadata.extraction_model,
            extraction_confidence=enhanced_metadata.extraction_confidence,
            processing_time_ms=enhanced_metadata.processing_time_ms,
            dot_file_created=(
                hash_ref.to_dot_file_name(request.file_path)
                if request.enable_dot_files
                else None
            ),
            database_stored=extractor.db_client is not None,
        )

        return UnifiedResponse(
            status="success",
            data=response_data.model_dump(),
            message=f"Enhanced metadata extracted with {len(enhanced_metadata.categories)} categories, "
            f"{len(enhanced_metadata.tags)} tags, {len(enhanced_metadata.entities)} entities",
            metadata={
                "operation": "enhanced_stamping",
                "langextract_model": enhanced_metadata.extraction_model,
                "processing_time_ms": enhanced_metadata.processing_time_ms,
            },
        )

    except Exception as e:
        logger.error(f"Enhanced stamping failed: {e}")
        return UnifiedResponse(
            status="error",
            error=str(e),
            message="Failed to perform enhanced metadata extraction",
        )


@enhanced_router.get("/metadata/{metadata_hash}", response_model=UnifiedResponse)
async def get_metadata_by_hash(
    metadata_hash: str,
    extractor: EnhancedMetadataExtractor = Depends(get_enhanced_extractor),
) -> UnifiedResponse:
    """Retrieve enhanced metadata by hash reference."""
    try:
        enhanced_metadata = await extractor.retrieve_metadata_by_hash(metadata_hash)

        if not enhanced_metadata:
            return UnifiedResponse(
                status="error",
                error="Metadata not found",
                message=f"No metadata found for hash: {metadata_hash[:12]}...",
            )

        response_data = MetadataQueryResponse(
            metadata_hash=metadata_hash,
            content_hash=enhanced_metadata.content_hash,
            enhanced_metadata=enhanced_metadata.model_dump(),
            created_at=enhanced_metadata.extraction_timestamp.isoformat(),
        )

        return UnifiedResponse(
            status="success",
            data=response_data.model_dump(),
            message=f"Enhanced metadata retrieved for hash: {metadata_hash[:12]}...",
            metadata={"operation": "metadata_retrieval"},
        )

    except Exception as e:
        logger.error(f"Failed to retrieve metadata: {e}")
        return UnifiedResponse(
            status="error", error=str(e), message="Failed to retrieve enhanced metadata"
        )


class HashExtractionRequest(BaseModel):
    """Request model for hash extraction from content."""

    content: str = Field(..., description="Content with embedded stamp")


@enhanced_router.post("/extract-hash", response_model=UnifiedResponse)
async def extract_hash_from_content(
    request: HashExtractionRequest,
    extractor: EnhancedMetadataExtractor = Depends(get_enhanced_extractor),
) -> UnifiedResponse:
    """Extract metadata hash from stamped content."""
    try:
        metadata_hash = extractor.extract_hash_from_stamp(request.content)

        if not metadata_hash:
            return UnifiedResponse(
                status="error",
                error="No stamp found",
                message="No metadata stamp found in the provided content",
            )

        return UnifiedResponse(
            status="success",
            data={"metadata_hash": metadata_hash, "hash_prefix": metadata_hash[:12]},
            message=f"Metadata hash extracted: {metadata_hash[:12]}...",
            metadata={"operation": "hash_extraction"},
        )

    except Exception as e:
        logger.error(f"Failed to extract hash: {e}")
        return UnifiedResponse(
            status="error", error=str(e), message="Failed to extract metadata hash"
        )


class QuickCategoriesRequest(BaseModel):
    """Request model for quick categories extraction."""

    content: str = Field(..., description="Content to categorize")
    file_path: Optional[str] = Field(None, description="Optional file path for context")


@enhanced_router.post("/quick-categories", response_model=UnifiedResponse)
async def extract_quick_categories(
    request: QuickCategoriesRequest,
    extractor: EnhancedMetadataExtractor = Depends(get_enhanced_extractor),
) -> UnifiedResponse:
    """Quick category extraction without full metadata processing."""
    try:
        categories = await extractor.get_content_categories(
            request.content, request.file_path
        )

        return UnifiedResponse(
            status="success",
            data={
                "categories": [cat.model_dump() for cat in categories],
                "category_count": len(categories),
            },
            message=f"Extracted {len(categories)} categories",
            metadata={"operation": "quick_categorization"},
        )

    except Exception as e:
        logger.error(f"Quick categorization failed: {e}")
        return UnifiedResponse(
            status="error", error=str(e), message="Failed to extract categories"
        )


class QuickTagsRequest(BaseModel):
    """Request model for quick tags extraction."""

    content: str = Field(..., description="Content to tag")
    file_path: Optional[str] = Field(None, description="Optional file path for context")


@enhanced_router.post("/quick-tags", response_model=UnifiedResponse)
async def extract_quick_tags(
    request: QuickTagsRequest,
    extractor: EnhancedMetadataExtractor = Depends(get_enhanced_extractor),
) -> UnifiedResponse:
    """Quick tag extraction without full metadata processing."""
    try:
        tags = await extractor.get_content_tags(request.content, request.file_path)

        return UnifiedResponse(
            status="success",
            data={
                "tags": [tag.model_dump() for tag in tags],
                "tag_count": len(tags),
                "tag_types": list({tag.tag_type for tag in tags}),
            },
            message=f"Extracted {len(tags)} tags",
            metadata={"operation": "quick_tagging"},
        )

    except Exception as e:
        logger.error(f"Quick tagging failed: {e}")
        return UnifiedResponse(
            status="error", error=str(e), message="Failed to extract tags"
        )


@enhanced_router.post("/analyze-file", response_model=UnifiedResponse)
async def analyze_uploaded_file(
    file: UploadFile = File(...),
    enable_langextract: bool = Query(True, description="Enable LangExtract analysis"),
    create_dot_file: bool = Query(
        False, description="Create dot file (requires file path)"
    ),
    extractor: EnhancedMetadataExtractor = Depends(get_enhanced_extractor),
) -> UnifiedResponse:
    """Analyze uploaded file content with enhanced metadata extraction."""
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode("utf-8")

        # Use filename as file path context
        file_path = file.filename or "uploaded_file"

        if enable_langextract:
            # Full enhanced analysis
            enhanced_metadata = await extractor.extract_enhanced_metadata(
                content=text_content,
                file_path=file_path,
                content_type=file.content_type,
            )

            return UnifiedResponse(
                status="success",
                data={
                    "file_info": {
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "size_bytes": len(content),
                    },
                    "enhanced_metadata": enhanced_metadata.model_dump(),
                    "categories": [
                        cat.model_dump() for cat in enhanced_metadata.categories
                    ],
                    "tags": [tag.model_dump() for tag in enhanced_metadata.tags],
                    "entities": [
                        entity.model_dump() for entity in enhanced_metadata.entities
                    ],
                },
                message=f"File analyzed with {len(enhanced_metadata.categories)} categories, "
                f"{len(enhanced_metadata.tags)} tags, {len(enhanced_metadata.entities)} entities",
                metadata={
                    "operation": "file_analysis",
                    "langextract_enabled": True,
                    "processing_time_ms": enhanced_metadata.processing_time_ms,
                },
            )
        else:
            # Basic analysis without LangExtract
            return UnifiedResponse(
                status="success",
                data={
                    "file_info": {
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "size_bytes": len(content),
                    },
                    "basic_analysis": {
                        "line_count": text_content.count("\n") + 1,
                        "character_count": len(text_content),
                        "word_count": len(text_content.split()),
                    },
                },
                message="Basic file analysis completed",
                metadata={"operation": "basic_file_analysis"},
            )

    except UnicodeDecodeError:
        return UnifiedResponse(
            status="error",
            error="File encoding not supported",
            message="File must be text-based for content analysis",
        )
    except Exception as e:
        logger.error(f"File analysis failed: {e}")
        return UnifiedResponse(
            status="error", error=str(e), message="Failed to analyze uploaded file"
        )


# Future API Enhancements (Phase 2):
# - Search endpoints for categories, tags, and entities
# - Batch processing endpoints for bulk operations
# - Metadata update/versioning endpoints for record management
