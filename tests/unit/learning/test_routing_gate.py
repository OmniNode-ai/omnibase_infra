# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RoutingGate unit tests (OMN-7404, Task 8 Phase 2).

RoutingGate is an audit-only gate per the reviewed/corrected plan spec
(docs/plans/2026-04-03-learning-infrastructure.md, Task 8 "Clarification
from review"): the classifier is a confidence estimator, never a model
selector. ``audit()`` must NEVER alter routing — it only reports
``{confidence, would_flag}`` for shadow-mode logging.

RED->GREEN: on the pre-implementation tree, ``omnibase_infra.learning.routing.gate``
does not exist, so every test here fails at collection (ImportError). After
``RoutingGate`` lands, all tests pass.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omnibase_infra.learning.routing.gate import RoutingGate


@pytest.mark.unit
class TestRoutingGateAudit:
    def test_audit_logs_agreement_when_confident(self) -> None:
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = 0.85
        gate = RoutingGate(classifier=mock_clf, confidence_threshold=0.75)
        audit = gate.audit(features={"prompt_length": 500, "utilization_score": 0.7})
        assert audit["confidence"] == pytest.approx(0.85)
        assert audit["would_flag"] is False  # high confidence = agrees with rule-based
        assert audit["audit_only"] is True

    def test_audit_flags_low_confidence_decision(self) -> None:
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = (
            0.25  # classifier thinks this is a bad decision
        )
        gate = RoutingGate(classifier=mock_clf, confidence_threshold=0.75)
        audit = gate.audit(features={"prompt_length": 500, "utilization_score": 0.7})
        assert audit["would_flag"] is True  # would flag, but does NOT alter routing
        assert audit["audit_only"] is True

    def test_audit_returns_noop_when_classifier_missing(self) -> None:
        gate = RoutingGate(classifier=None, confidence_threshold=0.75)
        audit = gate.audit(features={"prompt_length": 500, "utilization_score": 0.7})
        assert audit["confidence"] is None
        assert audit["would_flag"] is False  # graceful degradation
        assert audit["audit_only"] is True

    def test_audit_degrades_gracefully_on_classifier_exception(self) -> None:
        mock_clf = MagicMock()
        mock_clf.predict_proba.side_effect = RuntimeError("model corrupt")
        gate = RoutingGate(classifier=mock_clf, confidence_threshold=0.75)
        audit = gate.audit(features={"prompt_length": 500, "utilization_score": 0.7})
        assert audit["confidence"] is None
        assert audit["would_flag"] is False
        assert audit["audit_only"] is True

    def test_default_confidence_threshold_is_0_75(self) -> None:
        gate = RoutingGate(classifier=None)
        assert gate.confidence_threshold == pytest.approx(0.75)

    def test_audit_never_returns_a_selection_key(self) -> None:
        """Contract guard: audit-only means no model-selection field ever leaks out."""
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = 0.9
        gate = RoutingGate(classifier=mock_clf, confidence_threshold=0.75)
        audit = gate.audit(features={"prompt_length": 10, "utilization_score": 0.1})
        assert set(audit.keys()) == {"confidence", "would_flag", "audit_only"}
