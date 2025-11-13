"""
deployment_receiver - Receives and deploys Docker containers on remote systems. Handles Docker image packages with authentication, loads images into Docker daemon, deploys containers with configuration, runs health checks, and publishes Kafka events.

Generated: 2025-10-25T17:13:51.307905+00:00
ONEX v2.0 Compliant
"""

from .node import NodeDeploymentReceiverEffect

__all__ = ["NodeDeploymentReceiverEffect"]
