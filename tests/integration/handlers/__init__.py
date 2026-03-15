# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for infrastructure handlers.

This package contains integration tests for handlers that require
remote infrastructure (PostgreSQL, Consul, Vault, etc.).

Tests are marked with @pytest.mark.integration and will be skipped
when the required infrastructure is not available.
"""
