# === OmniNode:Tool_Metadata ===
# metadata_version: 0.1
# name: enhanced_metadata_extractor
# title: Enhanced Metadata Extractor with LangExtract
# version: 0.1.0
# namespace: omninode.services.metadata
# category: service.infrastructure.extraction
# kind: service
# role: metadata_intelligence
# description: |
#   Advanced metadata extraction using LangExtract for intelligent categorization,
#   tagging, and content analysis with O.N.E. v0.1 protocol compliance.
# tags: [langextract, metadata, categories, tags, intelligence, extraction]
# author: OmniNode Development Team
# license: MIT
# protocols_supported: [O.N.E. v0.1]
# runtime_constraints: {sandboxed: false, privileged: false, requires_network: true, requires_gpu: false}
# dependencies: [{"name": "langextract", "version": "^1.0.7"}, {"name": "pydantic", "version": "^2.11.7"}]
# environment: [python>=3.12]
# === /OmniNode:Tool_Metadata ===

"""Enhanced metadata extraction using LangExtract for intelligent content analysis.

This module provides advanced metadata extraction capabilities that go beyond
basic file information to include intelligent categorization, tagging, and
content analysis using LangExtract's LLM-powered extraction capabilities.
"""

import logging
import textwrap
from datetime import UTC, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

try:
    import langextract as lx

    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    lx = None

logger = logging.getLogger(__name__)


class ContentCategory(BaseModel):
    """Represents a content category with confidence score."""

    category: str = Field(..., description="Category name")
    subcategory: Optional[str] = Field(None, description="Subcategory if applicable")
    confidence: float = Field(..., description="Confidence score 0-1")
    reasoning: Optional[str] = Field(None, description="Why this category was assigned")
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Additional attributes"
    )


class ContentTag(BaseModel):
    """Represents a content tag with metadata."""

    tag: str = Field(..., description="Tag name")
    tag_type: str = Field(..., description="Type of tag (topic, keyword, entity, etc.)")
    relevance: float = Field(..., description="Relevance score 0-1")
    source_text: Optional[str] = Field(None, description="Text that generated this tag")
    position: Optional[tuple[int, int]] = Field(
        None, description="Character positions in source"
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Tag-specific attributes"
    )


class EntityExtraction(BaseModel):
    """Represents an extracted entity with rich metadata."""

    entity_text: str = Field(..., description="The extracted entity text")
    entity_type: str = Field(
        ..., description="Type of entity (person, place, concept, etc.)"
    )
    position: Optional[tuple[int, int]] = Field(None, description="Character positions")
    confidence: float = Field(..., description="Extraction confidence 0-1")
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Entity-specific attributes"
    )
    relationships: list[str] = Field(
        default_factory=list, description="Related entities"
    )


class EnhancedMetadata(BaseModel):
    """Complete enhanced metadata structure."""

    # Core fields
    extraction_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content_hash: str = Field(..., description="BLAKE3 hash of analyzed content")

    # Intelligence fields
    categories: list[ContentCategory] = Field(default_factory=list)
    tags: list[ContentTag] = Field(default_factory=list)
    entities: list[EntityExtraction] = Field(default_factory=list)

    # Content analysis
    content_type_detected: Optional[str] = Field(
        None, description="AI-detected content type"
    )
    language_detected: Optional[str] = Field(None, description="Detected language")
    sentiment: Optional[dict[str, float]] = Field(
        None, description="Sentiment analysis"
    )
    complexity_score: Optional[float] = Field(
        None, description="Content complexity 0-1"
    )

    # Relationships and context
    related_concepts: list[str] = Field(default_factory=list)
    key_themes: list[str] = Field(default_factory=list)
    topic_clusters: dict[str, list[str]] = Field(default_factory=dict)

    # Extraction metadata
    extraction_model: Optional[str] = Field(None, description="LLM model used")
    extraction_confidence: float = Field(
        0.0, description="Overall extraction confidence"
    )
    processing_time_ms: Optional[float] = Field(None, description="Processing time")


class MetadataHashReference(BaseModel):
    """Lightweight hash reference for clean file stamping."""

    metadata_hash: str = Field(..., description="BLAKE3 hash of the enhanced metadata")
    content_hash: str = Field(..., description="BLAKE3 hash of the original content")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str = Field(default="0.1", description="Metadata schema version")

    def to_lightweight_stamp(self) -> str:
        """Generate minimal stamp for embedding in files."""
        return f"<!-- ONEX:META:{self.metadata_hash} -->"

    def to_dot_file_name(self, file_path: str) -> str:
        """Generate dot file name for external metadata storage."""
        import os

        dir_path = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        return os.path.join(dir_path, f".{base_name}.onex-meta.json")


class EnhancedMetadataExtractor:
    """Advanced metadata extractor using LangExtract with hash-only stamping architecture.

    This implementation follows the omnibase_3 pattern of clean file stamping:
    1. Only embed lightweight hash references in files
    2. Store full enhanced metadata externally (database + dot files)
    3. Maintain idempotency and clean version control
    """

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        enable_visualization: bool = True,
        enable_dot_files: bool = True,
        extraction_confidence_threshold: float = 0.7,
        database_client=None,
    ):
        """Initialize the enhanced metadata extractor.

        Args:
            model_id: LLM model to use for extraction
            api_key: API key for the LLM provider
            enable_visualization: Whether to generate visualizations
            enable_dot_files: Whether to create .onex-meta.json files
            extraction_confidence_threshold: Minimum confidence for extractions
            database_client: Database client for metadata storage
        """
        if not LANGEXTRACT_AVAILABLE:
            raise ImportError(
                "LangExtract is not available. Install it with: pip install langextract"
            )

        self.model_id = model_id
        self.api_key = api_key
        self.enable_visualization = enable_visualization
        self.enable_dot_files = enable_dot_files
        self.confidence_threshold = extraction_confidence_threshold
        self.db_client = database_client

        # Define extraction prompts and examples
        self._setup_extraction_templates()

        logger.info(f"Enhanced metadata extractor initialized with model: {model_id}")
        logger.info(f"Hash-only stamping enabled, dot files: {enable_dot_files}")

    def _setup_extraction_templates(self):
        """Set up extraction prompts and examples for different content types."""

        # General content analysis prompt
        self.content_analysis_prompt = textwrap.dedent(
            """
            Analyze the provided content and extract:
            1. Categories and subcategories (with confidence scores)
            2. Relevant tags and keywords
            3. Named entities (people, places, organizations, concepts)
            4. Key themes and topics
            5. Content type and characteristics

            Use exact text for extractions. Provide confidence scores and reasoning.
            Group related extractions using attributes for better organization.
        """
        )

        # Code analysis prompt
        self.code_analysis_prompt = textwrap.dedent(
            """
            Analyze the provided code and extract:
            1. Programming language and frameworks
            2. Function names, class names, and variables
            3. Code patterns and architectural styles
            4. Dependencies and imports
            5. Code complexity and quality indicators
            6. Documentation and comments

            Focus on extracting meaningful metadata for code organization and discovery.
        """
        )

        # Document analysis prompt
        self.document_analysis_prompt = textwrap.dedent(
            """
            Analyze the provided document and extract:
            1. Document type and format
            2. Key topics and subjects
            3. Important entities and concepts
            4. Document structure elements
            5. Language and writing style
            6. Target audience indicators

            Provide rich metadata for document classification and retrieval.
        """
        )

        # Example data for training the extraction
        self.general_examples = [
            lx.data.ExampleData(
                text="This Python API service handles user authentication with JWT tokens and OAuth2 integration.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="category",
                        extraction_text="API service",
                        attributes={
                            "type": "software",
                            "confidence": 0.95,
                            "subcategory": "backend",
                        },
                    ),
                    lx.data.Extraction(
                        extraction_class="tag",
                        extraction_text="authentication",
                        attributes={"tag_type": "feature", "relevance": 0.9},
                    ),
                    lx.data.Extraction(
                        extraction_class="tag",
                        extraction_text="Python",
                        attributes={"tag_type": "language", "relevance": 0.95},
                    ),
                    lx.data.Extraction(
                        extraction_class="entity",
                        extraction_text="JWT tokens",
                        attributes={"entity_type": "technology", "confidence": 0.9},
                    ),
                    lx.data.Extraction(
                        extraction_class="entity",
                        extraction_text="OAuth2",
                        attributes={"entity_type": "protocol", "confidence": 0.9},
                    ),
                ],
            )
        ]

    async def extract_and_store_metadata(
        self,
        content: str,
        file_path: str,
        content_type: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> tuple[MetadataHashReference, EnhancedMetadata]:
        """Extract enhanced metadata and store externally with hash-only stamping.

        This is the main method following the omnibase_3 pattern:
        1. Extract enhanced metadata using LangExtract
        2. Store full metadata in database and/or dot files
        3. Return lightweight hash reference for file stamping

        Args:
            content: Content to analyze
            file_path: File path (required for dot file creation)
            content_type: Optional content type hint
            custom_prompt: Custom extraction prompt

        Returns:
            Tuple of (hash_reference, full_metadata)
        """
        # Extract the enhanced metadata
        enhanced_metadata = await self.extract_enhanced_metadata(
            content, file_path, content_type, custom_prompt
        )

        # Generate metadata hash for reference
        metadata_json = enhanced_metadata.model_dump_json()
        metadata_hash = self._generate_content_hash(metadata_json)

        # Create hash reference
        hash_ref = MetadataHashReference(
            metadata_hash=metadata_hash, content_hash=enhanced_metadata.content_hash
        )

        # Store metadata externally
        await self._store_metadata_externally(hash_ref, enhanced_metadata, file_path)

        logger.info(
            f"Enhanced metadata stored externally with hash: {metadata_hash[:12]}..."
        )
        return hash_ref, enhanced_metadata

    async def _store_metadata_externally(
        self,
        hash_ref: MetadataHashReference,
        metadata: EnhancedMetadata,
        file_path: str,
    ):
        """Store metadata in database and optionally create dot files."""

        # Store in database if available
        if self.db_client:
            try:
                await self._store_in_database(hash_ref, metadata)
                logger.debug(
                    f"Metadata stored in database: {hash_ref.metadata_hash[:12]}"
                )
            except Exception as e:
                logger.warning(f"Failed to store metadata in database: {e}")

        # Create dot file if enabled
        if self.enable_dot_files:
            try:
                await self._create_dot_file(hash_ref, metadata, file_path)
                logger.debug(
                    f"Dot file created: {hash_ref.to_dot_file_name(file_path)}"
                )
            except Exception as e:
                logger.warning(f"Failed to create dot file: {e}")

    async def _store_in_database(
        self, hash_ref: MetadataHashReference, metadata: EnhancedMetadata
    ):
        """Store enhanced metadata in database."""
        if not self.db_client:
            return

        # Convert to database format
        db_data = {
            "metadata_hash": hash_ref.metadata_hash,
            "content_hash": hash_ref.content_hash,
            "enhanced_metadata": metadata.model_dump(),
            "extraction_timestamp": metadata.extraction_timestamp,
            "extraction_model": metadata.extraction_model,
            "extraction_confidence": metadata.extraction_confidence,
        }

        # Store using existing database client
        # This would integrate with the existing MetadataStampingPostgresClient
        # Note: Requires enhanced_metadata table in database schema (Phase 2)
        # Schema should include: metadata_hash, categories, tags, entities, semantic_embedding
        pass

    async def _create_dot_file(
        self,
        hash_ref: MetadataHashReference,
        metadata: EnhancedMetadata,
        file_path: str,
    ):
        """Create dot file with enhanced metadata."""
        import json
        import os

        dot_file_path = hash_ref.to_dot_file_name(file_path)

        # Prepare dot file content
        dot_file_content = {
            "onex_metadata_version": "0.1",
            "hash_reference": hash_ref.model_dump(),
            "enhanced_metadata": metadata.model_dump(),
            "generated_by": "enhanced_metadata_extractor",
            "extraction_model": metadata.extraction_model,
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(dot_file_path), exist_ok=True)

        # Write dot file
        with open(dot_file_path, "w", encoding="utf-8") as f:
            json.dump(dot_file_content, f, indent=2, ensure_ascii=False, default=str)

    async def retrieve_metadata_by_hash(
        self, metadata_hash: str
    ) -> Optional[EnhancedMetadata]:
        """Retrieve enhanced metadata by hash from database or dot files.

        Args:
            metadata_hash: Hash of the metadata to retrieve

        Returns:
            Enhanced metadata if found, None otherwise
        """
        # Try database first
        if self.db_client:
            try:
                db_result = await self._retrieve_from_database(metadata_hash)
                if db_result:
                    return EnhancedMetadata.model_validate(
                        db_result["enhanced_metadata"]
                    )
            except Exception as e:
                logger.warning(f"Failed to retrieve from database: {e}")

        # Fallback to scanning dot files (less efficient)
        # This is a backup method - in practice, database should be primary
        logger.debug(f"Metadata not found in database for hash: {metadata_hash[:12]}")
        return None

    async def _retrieve_from_database(
        self, metadata_hash: str
    ) -> Optional[dict[str, Any]]:
        """Retrieve metadata from database by hash."""
        if not self.db_client:
            return None

        # Query database for enhanced metadata (Phase 2 implementation)
        # Implementation placeholder:
        #   query = "SELECT * FROM enhanced_metadata WHERE metadata_hash = $1"
        #   result = await self.db_client.fetchrow(query, metadata_hash)
        #   return EnhancedMetadata(**result) if result else None
        return None

    def extract_hash_from_stamp(self, stamped_content: str) -> Optional[str]:
        """Extract metadata hash from lightweight stamp in content.

        Args:
            stamped_content: Content that may contain lightweight stamps

        Returns:
            Metadata hash if found, None otherwise
        """
        import re

        # Look for lightweight stamp pattern
        pattern = r"<!-- ONEX:META:([a-f0-9]{64}) -->"
        match = re.search(pattern, stamped_content)

        if match:
            return match.group(1)

        # Also check for legacy patterns
        legacy_patterns = [
            r"<!-- ONEX_METADATA_START -->.*?hash: ([a-f0-9]{64})",
            r"ONEX_HASH:([a-f0-9]{64})",
        ]

        for pattern in legacy_patterns:
            match = re.search(pattern, stamped_content, re.DOTALL)
            if match:
                return match.group(1)

        return None

    async def get_content_categories(
        self, content: str, file_path: Optional[str] = None
    ) -> list[ContentCategory]:
        """Quick category extraction without full metadata processing.

        Args:
            content: Content to categorize
            file_path: Optional file path for context

        Returns:
            List of content categories
        """
        enhanced_metadata = await self.extract_enhanced_metadata(content, file_path)
        return enhanced_metadata.categories

    async def get_content_tags(
        self, content: str, file_path: Optional[str] = None
    ) -> list[ContentTag]:
        """Quick tag extraction without full metadata processing.

        Args:
            content: Content to analyze
            file_path: Optional file path for context

        Returns:
            List of content tags
        """
        enhanced_metadata = await self.extract_enhanced_metadata(content, file_path)
        return enhanced_metadata.tags

    async def extract_enhanced_metadata(
        self,
        content: str,
        file_path: Optional[str] = None,
        content_type: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> EnhancedMetadata:
        """Extract enhanced metadata from content using LangExtract.

        Args:
            content: Content to analyze
            file_path: Optional file path for context
            content_type: Optional content type hint
            custom_prompt: Custom extraction prompt

        Returns:
            Enhanced metadata with categories, tags, entities, and more
        """
        start_time = datetime.now()

        try:
            # Select appropriate prompt based on content type
            prompt = self._select_prompt(content, content_type, custom_prompt)
            examples = self._select_examples(content_type)

            # Perform LangExtract extraction
            result = lx.extract(
                text_or_documents=content,
                prompt_description=prompt,
                examples=examples,
                model_id=self.model_id,
                api_key=self.api_key,
            )

            # Process extraction results
            enhanced_metadata = await self._process_extraction_results(
                result, content, file_path, start_time
            )

            # Generate visualization if enabled
            if self.enable_visualization:
                await self._generate_visualization(result, enhanced_metadata)

            logger.info(
                f"Enhanced metadata extracted: {len(enhanced_metadata.categories)} categories, "
                f"{len(enhanced_metadata.tags)} tags, {len(enhanced_metadata.entities)} entities"
            )

            return enhanced_metadata

        except Exception as e:
            logger.error(f"Error extracting enhanced metadata: {e}")
            # Return basic metadata on failure
            return EnhancedMetadata(
                content_hash=self._generate_content_hash(content),
                extraction_confidence=0.0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    def _select_prompt(
        self, content: str, content_type: Optional[str], custom_prompt: Optional[str]
    ) -> str:
        """Select appropriate extraction prompt based on content analysis."""
        if custom_prompt:
            return custom_prompt

        # Analyze content to determine type
        content_lower = content.lower()

        if (content_type and "code" in content_type.lower()) or any(
            keyword in content_lower
            for keyword in ["def ", "class ", "import ", "function", "var ", "const "]
        ):
            return self.code_analysis_prompt
        elif any(
            keyword in content_lower
            for keyword in ["# ", "## ", "### ", "title:", "author:"]
        ):
            return self.document_analysis_prompt
        else:
            return self.content_analysis_prompt

    def _select_examples(self, content_type: Optional[str]) -> list:
        """Select appropriate examples based on content type."""
        # For now, return general examples
        # Future enhancement: Add content-type-specific examples
        # (code: function/class extraction, docs: topic modeling, media: visual analysis)
        return self.general_examples

    async def _process_extraction_results(
        self, result, content: str, file_path: Optional[str], start_time: datetime
    ) -> EnhancedMetadata:
        """Process LangExtract results into structured metadata."""

        categories = []
        tags = []
        entities = []

        for extraction in result.extractions:
            extraction_data = {
                "source_text": extraction.extraction_text,
                "position": None,
                "confidence": 0.8,  # Default confidence
                "attributes": extraction.attributes or {},
            }

            # Extract position if available
            if extraction.char_interval:
                extraction_data["position"] = (
                    extraction.char_interval.start_pos,
                    extraction.char_interval.end_pos,
                )

            # Extract confidence from attributes
            if "confidence" in extraction_data["attributes"]:
                extraction_data["confidence"] = float(
                    extraction_data["attributes"]["confidence"]
                )

            # Categorize extractions
            if extraction.extraction_class == "category":
                category = ContentCategory(
                    category=extraction.extraction_text,
                    subcategory=extraction_data["attributes"].get("subcategory"),
                    confidence=extraction_data["confidence"],
                    reasoning=extraction_data["attributes"].get("reasoning"),
                    attributes=extraction_data["attributes"],
                )
                categories.append(category)

            elif extraction.extraction_class == "tag":
                tag = ContentTag(
                    tag=extraction.extraction_text,
                    tag_type=extraction_data["attributes"].get("tag_type", "general"),
                    relevance=extraction_data["attributes"].get(
                        "relevance", extraction_data["confidence"]
                    ),
                    source_text=extraction_data["source_text"],
                    position=extraction_data["position"],
                    attributes=extraction_data["attributes"],
                )
                tags.append(tag)

            elif extraction.extraction_class == "entity":
                entity = EntityExtraction(
                    entity_text=extraction.extraction_text,
                    entity_type=extraction_data["attributes"].get(
                        "entity_type", "unknown"
                    ),
                    position=extraction_data["position"],
                    confidence=extraction_data["confidence"],
                    attributes=extraction_data["attributes"],
                )
                entities.append(entity)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Calculate overall confidence
        all_confidences = (
            [cat.confidence for cat in categories]
            + [tag.relevance for tag in tags]
            + [entity.confidence for entity in entities]
        )
        avg_confidence = (
            sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        )

        return EnhancedMetadata(
            content_hash=self._generate_content_hash(content),
            categories=categories,
            tags=tags,
            entities=entities,
            extraction_model=self.model_id,
            extraction_confidence=avg_confidence,
            processing_time_ms=processing_time,
        )

    async def _generate_visualization(self, result, metadata: EnhancedMetadata):
        """Generate interactive visualization of extraction results."""
        if not self.enable_visualization:
            return

        try:
            # Save results for visualization
            output_name = f"enhanced_metadata_{metadata.content_hash[:8]}"
            lx.io.save_annotated_documents(
                [result], output_name=f"{output_name}.jsonl", output_dir="."
            )

            # Generate HTML visualization
            html_content = lx.visualize(f"{output_name}.jsonl")
            with open(f"{output_name}_visualization.html", "w") as f:
                if hasattr(html_content, "data"):
                    f.write(html_content.data)
                else:
                    f.write(html_content)

            logger.info(f"Visualization saved: {output_name}_visualization.html")

        except Exception as e:
            logger.warning(f"Failed to generate visualization: {e}")

    def _generate_content_hash(self, content: str) -> str:
        """Generate BLAKE3 hash of content."""
        import blake3

        hasher = blake3.blake3()
        hasher.update(content.encode("utf-8"))
        return hasher.hexdigest()

    async def extract_for_file_types(
        self, content: str, file_extension: str, **kwargs
    ) -> EnhancedMetadata:
        """Extract metadata optimized for specific file types.

        Args:
            content: File content
            file_extension: File extension (e.g., '.py', '.md', '.json')
            **kwargs: Additional extraction parameters

        Returns:
            Enhanced metadata optimized for the file type
        """
        # Map file extensions to content types and specialized prompts
        file_type_mapping = {
            # Code files
            ".py": ("code", self.code_analysis_prompt),
            ".js": ("code", self.code_analysis_prompt),
            ".ts": ("code", self.code_analysis_prompt),
            ".java": ("code", self.code_analysis_prompt),
            ".cpp": ("code", self.code_analysis_prompt),
            ".rs": ("code", self.code_analysis_prompt),
            ".go": ("code", self.code_analysis_prompt),
            # Documentation
            ".md": ("document", self.document_analysis_prompt),
            ".rst": ("document", self.document_analysis_prompt),
            ".txt": ("document", self.document_analysis_prompt),
            # Configuration
            ".json": ("config", self.content_analysis_prompt),
            ".yaml": ("config", self.content_analysis_prompt),
            ".yml": ("config", self.content_analysis_prompt),
            ".toml": ("config", self.content_analysis_prompt),
            ".ini": ("config", self.content_analysis_prompt),
        }

        content_type, prompt = file_type_mapping.get(
            file_extension.lower(), ("general", self.content_analysis_prompt)
        )

        return await self.extract_enhanced_metadata(
            content=content, content_type=content_type, custom_prompt=prompt, **kwargs
        )
