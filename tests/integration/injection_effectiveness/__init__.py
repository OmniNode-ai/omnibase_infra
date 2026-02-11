# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Integration tests for the injection effectiveness golden path pipeline.

This package validates the end-to-end injection effectiveness flow:
- WriterInjectionEffectivenessPostgres metric writes to PostgreSQL
- LedgerSinkInjectionEffectivenessPostgres ledger entry creation
- Correlation ID traceability across both metric and ledger write paths

Related tickets: OMN-2170, OMN-2078
"""
