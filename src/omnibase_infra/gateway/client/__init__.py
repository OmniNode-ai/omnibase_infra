# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Client half of the gateway attach cycle -- what ``onex auth`` drives (OMN-15922).

The server half of this contract lives in ``nodes/node_gateway_attach_effect``;
this package is the caller that holds a credential, mints a token against it,
attaches, and re-attaches before its session ceiling. The two halves share the
session and renewal models rather than mirroring them, so the contract has one
definition and cannot drift.
"""
