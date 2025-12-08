# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for Docker infrastructure.

These tests require a running Docker daemon and validate:
- Docker image build process
- Container runtime behavior
- Security properties (non-root execution)
- Health check functionality
- Resource limits and graceful shutdown
"""
