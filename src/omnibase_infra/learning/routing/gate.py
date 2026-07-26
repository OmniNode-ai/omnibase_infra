# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Audit-only routing confidence gate (OMN-7404, Phase 2 Task 8).

``RoutingGate`` wraps an optional routing-decision confidence classifier and
reports whether the rule-based router's decision looks confident, WITHOUT
ever selecting a model or altering routing behavior.

Source: docs/plans/2026-04-03-learning-infrastructure.md, Task 8. The plan's
"Clarification from review" corrects the original override-capable design:

    The classifier predicts "is this routing decision likely good?" — a
    binary confidence estimator, NOT a model selector. Shadow-mode only in
    Phase 2: the audit result is logged, but never acts on routing.

Consequently this module implements ``audit()`` (never ``recommend()`` /
``use_classifier`` override semantics some earlier ticket drafts described)
and never mutates or gates the `~ModelRoutingDecision` produced by
`HandlerScoreModels` — see `handler_score_models.py` for the shadow-mode
wiring.

This class is a pure, dependency-free unit: no file I/O, no pickle/pandas
dependency. Loading a persisted classifier artifact (Task 7, not yet built —
no `RoutingClassifier` exists anywhere in this repo as of OMN-7404) is
explicitly OUT of scope here and belongs to whatever constructs the
`RoutingGate` (a DI/registry seam), never to the gate or the COMPUTE handler
itself — COMPUTE nodes must remain I/O-free and deterministic (CLAUDE.md
§7a). Graceful degradation is therefore expressed as "classifier is `None`",
not as a file-existence check inside a compute path.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from omnibase_infra.learning.routing.typed_dict_routing_audit import (
    TypedDictRoutingAudit,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIDENCE_THRESHOLD = 0.75


@runtime_checkable
class ProtocolRoutingClassifier(Protocol):
    """Minimal duck-typed interface a routing confidence classifier must expose.

    Deliberately independent of any particular ML library (no pandas/sklearn
    dependency) since Task 7 (train + persist ``RoutingClassifier``) has not
    been built. Any object exposing ``predict_proba(features) -> float`` in
    [0.0, 1.0] satisfies this protocol.
    """

    def predict_proba(self, features: dict[str, object]) -> float:
        """Return P(this routing decision is a good one) in [0.0, 1.0]."""
        ...


class RoutingGate:
    """Audit-only gate. Evaluates routing decisions but does NOT alter them.

    Phase 2: shadow mode only — the caller logs/stores the audit result for
    later disagreement-rate analysis. Graduation to a veto-capable mode is an
    explicit future decision (plan Task 8 note) gated on >=500 audited
    decisions and is NOT implemented here.
    """

    def __init__(
        self,
        classifier: ProtocolRoutingClassifier | None,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._clf = classifier
        self._threshold = confidence_threshold

    @property
    def confidence_threshold(self) -> float:
        return self._threshold

    def audit(self, features: dict[str, object]) -> TypedDictRoutingAudit:
        """Audit a routing decision. Returns confidence + flag, never alters routing.

        Returns:
            ``{"confidence": float | None, "would_flag": bool, "audit_only": True}``
            ``would_flag`` is True when the classifier's confidence that the
            rule-based decision was good falls BELOW ``confidence_threshold``.
        """
        if self._clf is None:
            return {"confidence": None, "would_flag": False, "audit_only": True}
        try:
            proba = float(self._clf.predict_proba(features))
        except Exception:  # noqa: BLE001 - classifier failure must never break routing
            logger.warning(
                "Routing classifier audit failed; degrading to no-op.", exc_info=True
            )
            return {"confidence": None, "would_flag": False, "audit_only": True}
        would_flag = proba < self._threshold
        return {"confidence": proba, "would_flag": would_flag, "audit_only": True}


__all__ = ["ProtocolRoutingClassifier", "RoutingGate"]
