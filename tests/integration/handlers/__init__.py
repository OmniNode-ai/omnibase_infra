# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for infrastructure handlers.

This package contains integration tests for handlers that require
remote infrastructure (PostgreSQL, Consul, Vault, etc.).

Tests are marked with @pytest.mark.integration and will be skipped
when the required infrastructure is not available.
"""
