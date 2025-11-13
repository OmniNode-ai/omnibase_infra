"""
Stamp Request Input Model for Bridge Orchestrator.

Input state model for stamping workflow requests.

ONEX Compliance:
- Suffix-based naming: ModelStampRequestInput
- Strong typing with Pydantic validation
- UUID correlation tracking
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelStampRequestInput(BaseModel):
    """
    Input model for stamping workflow requests.

    Represents a request from MetadataStampingService to create
    a cryptographic stamp for file content.
    """

    # Required fields
    file_path: str = Field(..., description="Path to file being stamped")
    file_content: bytes = Field(..., description="Raw file content for hashing")
    content_type: str = Field(..., description="MIME type of content")

    # Optional O.N.E. v0.1 compliance fields
    namespace: str = Field(
        default="omninode.services.metadata",
        description="Namespace for multi-tenant organization",
    )
    op_id: UUID = Field(
        default_factory=uuid4, description="Operation ID for correlation tracking"
    )

    # Workflow configuration
    enable_onextree_intelligence: bool = Field(
        default=False, description="Whether to route to OnexTree for AI analysis"
    )
    intelligence_context: Optional[str] = Field(
        default=None, description="Context for OnexTree intelligence analysis"
    )

    # Metadata
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Additional metadata for stamping"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Request timestamp (UTC)"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "file_path": "/data/documents/contract.pdf",
                "file_content": "base64_encoded_pdf_content_here",
                "content_type": "application/pdf",
                "namespace": "omninode.services.metadata",
                "enable_onextree_intelligence": True,
                "intelligence_context": "legal_document_validation",
                "metadata": {"project": "legal_review", "client": "acme_corp"},
            }
        },
    )
