# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What a declared ``Gate:`` form resolves to (OMN-18414)."""

from __future__ import annotations

from enum import Enum


class EnumGateBindingProbe(str, Enum):
    """Whether a form is a proof pointer or a traceability binding.

    ``WORKFLOW_RUN`` is a proof pointer: the form names a workflow whose newest
    completed run's conclusion is the ticket's own declared evidence, and
    anything but success holds the flip.

    ``NONE`` is a traceability binding: the form names which commitment a
    ticket exists to serve. It is not weaker evidence, it is a different kind
    of statement, and the contract's ``$comment`` records why resolving the
    remaining forms to a live probe is refuted rather than merely
    unimplemented -- the beta board is generated FROM Linear, so probing it to
    decide a Linear flip is a cycle, and a live-gate defect names a surface
    that is red by construction at filing time. A ``NONE`` form can never
    produce a gate-probe hold.
    """

    WORKFLOW_RUN = "workflow_run"
    NONE = "none"
