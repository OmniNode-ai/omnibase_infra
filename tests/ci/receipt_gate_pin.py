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
RECEIPT_GATE_PIN_TAG = "dev-20261030152335-66d35d2283c3"
RECEIPT_GATE_PIN_SHA = "66d35d2283c3011c436799eacb9bce1a434ee6c9"
