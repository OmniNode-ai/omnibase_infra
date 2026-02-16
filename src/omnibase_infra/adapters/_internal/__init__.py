# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Internal adapters - not for direct use outside handlers and tests.

Architecture Rule (OMN-2286):
    Adapters in this package wrap external SDKs and provide a thin, testable
    interface. They MUST NOT be used directly by application code. All access
    MUST go through the corresponding handler, which owns caching, circuit
    breaking, and audit concerns.

    Direct adapter usage outside of handlers and their tests will be flagged
    by the ``validator_no_direct_adapter`` architecture rule.
"""

__all__: list[str] = []
