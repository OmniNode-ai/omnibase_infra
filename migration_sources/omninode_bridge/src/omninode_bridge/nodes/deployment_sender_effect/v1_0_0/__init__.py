"""
deployment_sender - Create deployment sender effect node for packaging and transferring Docker containers to remote systems.
Builds Docker images, creates compressed packages with BLAKE3 checksums, and transfers via HTTP/rsync to remote receivers.
Publishes Kafka events for lifecycle tracking.
Implements io_operations: package_container (build/export/compress/checksum), transfer_package (validate/upload/verify), publish_transfer_event.
Performance: <20s image build, <10s transfer for 1GB packages.

Generated: 2025-10-25T17:15:55.709570+00:00
ONEX v2.0 Compliant
"""

from .node import NodeDeploymentSenderEffect

__all__ = ["NodeDeploymentSenderEffect"]
