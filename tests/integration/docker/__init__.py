# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for Docker infrastructure.

These tests require a running Docker daemon and validate:
- Docker image build process
- Container runtime behavior
- Security properties (non-root execution)
- Health check functionality
- Resource limits and graceful shutdown
"""
