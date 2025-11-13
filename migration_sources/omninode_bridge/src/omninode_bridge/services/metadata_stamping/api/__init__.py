"""API components for metadata stamping service."""

from fastapi import APIRouter

from .registry import router as registry_router
from .router import router as main_router

# Create combined router
router = APIRouter()
router.include_router(main_router)
router.include_router(registry_router)

__all__ = ["router"]
