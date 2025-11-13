"""
Standalone REST API for NodeLLMEffect - Simplified version without omnibase runtime.

This is a minimal implementation for demo/bridge environments.
Provides a REST API wrapper for LLM API calls supporting multiple tiers.
"""

import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Try to import the real node, fall back to None if omnibase_core is not available
try:
    from omnibase_core.models.core import ModelContainer

    from .node import NodeLLMEffect
except ImportError:
    NodeLLMEffect = None  # type: ignore
    ModelContainer = None  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global node instance (will be initialized on startup)
node_instance: Optional["NodeLLMEffect"] = None


# Configuration model for standalone mode
class StandaloneConfig(BaseModel):
    """Configuration for standalone LLM Effect node mode."""

    zai_api_key: str
    zai_endpoint: str
    default_tier: str = "CLOUD_FAST"
    max_tokens: int = 4000
    temperature: float = 0.7
    environment: str = "development"


# Pydantic models for API
class LLMGenerationRequest(BaseModel):
    """Request model for LLM generation."""

    prompt: str = Field(..., description="The prompt to send to the LLM")
    tier: str = Field(
        default="CLOUD_FAST",
        description="LLM tier to use (LOCAL, CLOUD_FAST, CLOUD_PREMIUM)",
    )
    max_tokens: int = Field(default=4000, description="Maximum tokens to generate")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    operation_type: str = Field(
        default="general",
        description="Type of operation (node_generation, general, etc.)",
    )


class LLMGenerationResponse(BaseModel):
    """Response model for LLM generation."""

    success: bool
    generated_text: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    tier_used: Optional[str] = None
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    mode: str
    node_initialized: bool


# Create FastAPI application
app = FastAPI(
    title="NodeLLMEffect API (Standalone)",
    description="Standalone REST API for ONEX v2.0 LLM integration",
    version="1.0.0",
)

# Configure CORS - Security: Use environment variable for allowed origins
# Default to localhost:8000 for development. Set ALLOWED_ORIGINS env var for production.
# Never use wildcard ("*") with allow_credentials=True as it's a security vulnerability.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Configurable origins from environment
    allow_credentials=True,
    allow_methods=["POST", "GET"],  # Only allow necessary HTTP methods
    allow_headers=["Content-Type", "Authorization"],  # Only allow necessary headers
)


@app.on_event("startup")
async def startup_event():
    """Initialize node on application startup."""
    global node_instance

    try:
        # Check for required environment variables
        zai_api_key = os.getenv("ZAI_API_KEY")
        zai_endpoint = os.getenv("ZAI_ENDPOINT", "https://api.z.ai/api/anthropic")

        if not zai_api_key:
            logger.warning("ZAI_API_KEY not set - node will run in degraded mode")
            node_instance = None
            return

        # Create standalone configuration from environment
        config = StandaloneConfig(
            zai_api_key=zai_api_key,
            zai_endpoint=zai_endpoint,
            default_tier=os.getenv("DEFAULT_LLM_TIER", "CLOUD_FAST"),
            max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", "4000")),
            temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
            environment=os.getenv("ENVIRONMENT", "development"),
        )

        logger.info("Standalone configuration loaded")

        # Initialize node if omnibase_core is available
        if NodeLLMEffect and ModelContainer:
            try:
                # SECURITY: Container should NOT contain secrets
                # NodeLLMEffect reads ZAI_API_KEY and ZAI_ENDPOINT from environment
                # variables internally (see node.py:137-138). Passing secrets through
                # containers risks exposure in logs, metrics, and debugging output.
                # Container is for non-sensitive configuration only.
                container = ModelContainer(
                    value={},  # Empty - all config read from environment by node
                    container_type="config",
                )
                node_instance = NodeLLMEffect(container)
                logger.info("NodeLLMEffect initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize node: {e}", exc_info=True)
                node_instance = None
        else:
            logger.warning("omnibase_core not available - running in degraded mode")
            node_instance = None

    except Exception as e:
        logger.error(f"Failed to initialize standalone mode: {e}", exc_info=True)
        node_instance = None


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown node and cleanup resources."""
    global node_instance

    if node_instance:
        try:
            # Node cleanup if needed
            logger.info("NodeLLMEffect shutdown complete")
        except Exception as e:
            logger.error(f"Error during node shutdown: {e}", exc_info=True)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if node_instance else "degraded",
        service="NodeLLMEffect",
        version="1.0.0",
        mode="standalone",
        node_initialized=node_instance is not None,
    )


@app.post(
    "/generate",
    response_model=LLMGenerationResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_text(request: LLMGenerationRequest):
    """
    Generate text using LLM.

    Args:
        request: LLM generation request

    Returns:
        LLMGenerationResponse with generated text and metadata
    """
    if not node_instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM node not initialized - check ZAI_API_KEY configuration",
        )

    try:
        # Import models
        from omnibase_core.enums.enum_node_type import EnumNodeType
        from omnibase_core.models.contracts.model_contract_effect import (
            ModelContractEffect,
        )
        from omnibase_core.models.contracts.model_io_operation_config import (
            ModelIOOperationConfig,
        )
        from omnibase_core.primitives.model_semver import ModelSemVer

        # Create contract for node execution
        contract = ModelContractEffect(
            name="llm_generation",
            version=ModelSemVer(major=1, minor=0, patch=0),
            description=f"Generate text for: {request.operation_type}",
            node_type=EnumNodeType.EFFECT,
            input_model="ModelLLMRequest",
            output_model="ModelLLMResponse",
            io_operations=[
                ModelIOOperationConfig(
                    operation_type="api_call",
                    operation_name="llm_generate",
                    requires_network=True,
                )
            ],
            input_state={
                "prompt": request.prompt,
                "tier": request.tier,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "operation_type": request.operation_type,
            },
        )

        # Execute node
        result = await node_instance.execute_effect(contract)

        # result is already a ModelLLMResponse object
        return LLMGenerationResponse(
            success=True,
            generated_text=result.generated_text,
            tokens_used=result.tokens_total,
            cost_usd=result.cost_usd,
            tier_used=(
                result.tier_used.value
                if hasattr(result.tier_used, "value")
                else str(result.tier_used)
            ),
            error_message=None,
        )

    except Exception as e:
        logger.error(f"Error during LLM generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "generated_text": None,
                "tokens_used": None,
                "cost_usd": None,
                "tier_used": request.tier,
                "error_message": str(e),
            },
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return {
        "node_initialized": node_instance is not None,
        "mode": "standalone",
    }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "NodeLLMEffect",
        "version": "1.0.0",
        "mode": "standalone",
        "status": "running",
        "node_initialized": node_instance is not None,
        "message": "Standalone REST API for LLM operations",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8070,
        log_level="info",
    )
