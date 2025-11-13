"""OmniNode Bridge models."""

# ruff: noqa: F401, I001

# Import database models conditionally
try:
    from sqlalchemy.orm import declarative_base

    # Create a base class for our database models
    Base = declarative_base()

    # Create metadata instance for Alembic
    metadata = Base.metadata

    # Import all database models to ensure they're registered with the metadata
    from .database_models import *  # noqa: F403

    _DATABASE_AVAILABLE = True
except ImportError:
    # Create placeholders for environments without SQLAlchemy
    Base = None
    metadata = None
    _DATABASE_AVAILABLE = False

# Workflow models are always available (no heavy dependencies)
try:
    from .workflow import (
        EventType,
        MatrixStrategy,
        PermissionLevel,
        PermissionSet,
        WorkflowConfig,
        WorkflowJob,
        WorkflowStep,
        WorkflowTemplate,
    )

    _WORKFLOW_AVAILABLE = True
except ImportError:
    _WORKFLOW_AVAILABLE = False

# Node registration models (Pydantic, always available)
try:
    from .node_registration import (
        ModelNodeRegistration,
        ModelNodeRegistrationCreate,
        ModelNodeRegistrationUpdate,
    )

    _NODE_REGISTRATION_AVAILABLE = True
except ImportError:
    _NODE_REGISTRATION_AVAILABLE = False

__all__ = []

# Add database models if available
if _DATABASE_AVAILABLE:
    __all__.extend(["Base", "metadata"])

# Add workflow models if available
if _WORKFLOW_AVAILABLE:
    __all__.extend(
        [
            "EventType",
            "MatrixStrategy",
            "PermissionLevel",
            "PermissionSet",
            "WorkflowConfig",
            "WorkflowJob",
            "WorkflowStep",
            "WorkflowTemplate",
        ]
    )

# Add node registration models if available
if _NODE_REGISTRATION_AVAILABLE:
    __all__.extend(
        [
            "ModelNodeRegistration",
            "ModelNodeRegistrationCreate",
            "ModelNodeRegistrationUpdate",
        ]
    )

# Intent publisher models (always available)
try:
    from .model_intent_publish_result import ModelIntentPublishResult

    __all__.append("ModelIntentPublishResult")
except ImportError:
    pass
