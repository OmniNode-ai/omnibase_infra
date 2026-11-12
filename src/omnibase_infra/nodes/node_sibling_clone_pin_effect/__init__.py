# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Node Sibling Clone Pin Effect: resolves the pin of a sibling clone (OMN-23947).

Kept free of imports on purpose: the workspace-build preflight loads the handler
with an interpreter that has pydantic and nothing else, so this package must not
pull in the node class (omnibase_core).
"""
