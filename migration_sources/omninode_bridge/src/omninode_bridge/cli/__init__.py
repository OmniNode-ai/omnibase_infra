"""
Command-line interface tools for OmniNode Bridge.

This module provides CLI utilities for interacting with the OmniNode Bridge
workflow coordinator and other services via event-driven interfaces.

Note: Imports are lazy to avoid circular dependencies and allow
independent use of submodules (e.g., codegen CLI).
"""

__all__ = [
    "WorkflowSubmissionCLI",
]


def __getattr__(name: str):
    """
    Lazy import of CLI tools to avoid circular dependencies.

    This allows independent use of CLI submodules (e.g., codegen)
    without triggering imports of unrelated components.
    """
    if name == "WorkflowSubmissionCLI":
        try:
            from .workflow_submit import WorkflowSubmissionCLI

            return WorkflowSubmissionCLI
        except ImportError:
            # Create a placeholder for CI environments
            class WorkflowSubmissionCLI:  # type: ignore[no-redef]
                """Placeholder for environments where workflow submission dependencies are not available."""

                def __init__(self) -> None:
                    raise ImportError(
                        "WorkflowSubmissionCLI requires additional dependencies (aiokafka, etc.). "
                        "Install with: pip install omninode-bridge[kafka]"
                    )

            return WorkflowSubmissionCLI

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
