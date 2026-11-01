# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The omnibase_core receipt-gate pin this repository's caller holds (OMN-20134).

These two constants are the only test-side copy of the pin. omnimarket's
node_receipt_gate_pin_advance_compute rewrites them in the same commit as the
caller's ``uses:`` line and the required-checks.yaml ``cross_repo_ref``, so a pin
advance never leaves a test asserting the previous pin.

The caller is a ``pull_request_target`` workflow that scripts/audit-runner-routing.py
holds to a full-SHA pin, so the line carries the commit sha with the dev-merge tag
named beside it; test_repo_evidence_caller_check_timeout_omn21433.py resolves that
tag through the GitHub API and requires it to point at the pinned sha.
"""

RECEIPT_GATE_REUSABLE = "OmniNode-ai/omnibase_core/.github/workflows/receipt-gate.yml"
RECEIPT_GATE_PIN_TAG = "dev-20261101005012-ac2f26e3e5bf"
RECEIPT_GATE_PIN_SHA = "ac2f26e3e5bfd02c92496f7d3a9a14071f2c4432"
