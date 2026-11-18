# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The omnibase_core receipt-gate pin this repository's caller holds (OMN-20134).

These two constants are the only test-side copy of the pin. omnimarket's
node_receipt_gate_pin_advance_compute rewrites them in the same commit as the
caller's ``uses:`` line and the required-checks.yaml ``cross_repo_ref``, so a pin
advance never leaves a test asserting the previous pin.

The caller runs on ``pull_request``. The runner-routing audit requires a full-SHA
pin, and the uses line names the corresponding dev-merge tag beside it.
test_repo_evidence_caller_check_timeout_omn21433.py resolves that tag and
requires it to point at the pinned sha.
"""

RECEIPT_GATE_REUSABLE = "OmniNode-ai/omnibase_core/.github/workflows/receipt-gate.yml"
RECEIPT_GATE_PIN_TAG = "dev-20261108023315-fe0d8082717e"
RECEIPT_GATE_PIN_SHA = "fe0d8082717eec6717f138825c0aad71efc27ee0"
