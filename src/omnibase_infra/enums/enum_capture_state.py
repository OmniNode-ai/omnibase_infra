# SPDX-FileCopyrightText: 2025 OmniNode Team <info@omninode.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Corpus capture state enum.

Defines the lifecycle states for corpus capture operations.

.. versionadded:: 0.5.0
    Added for CorpusCapture (OMN-1203)
"""

from enum import Enum


class EnumCaptureState(str, Enum):
    """
    Lifecycle states for corpus capture.

    The capture service transitions through these states during its lifecycle:

    State Transitions::

        IDLE ──create_corpus()──> READY
        READY ──start_capture()──> CAPTURING
        CAPTURING ──pause_capture()──> PAUSED
        CAPTURING ──max_executions reached──> FULL
        PAUSED ──resume_capture()──> CAPTURING
        PAUSED ──close_corpus()──> CLOSED
        CAPTURING ──close_corpus()──> CLOSED
        FULL ──close_corpus()──> CLOSED
        READY ──close_corpus()──> CLOSED

    .. versionadded:: 0.5.0
        Added for CorpusCapture (OMN-1203)
    """

    IDLE = "idle"
    """No active corpus. Initial state or after corpus is archived."""

    READY = "ready"
    """Corpus created and configured, but not yet capturing."""

    CAPTURING = "capturing"
    """Actively capturing executions into the corpus."""

    PAUSED = "paused"
    """Capture temporarily suspended. Can be resumed."""

    FULL = "full"
    """Corpus has reached max_executions limit. Can only close."""

    CLOSED = "closed"
    """Corpus finalized and sealed. No more captures allowed."""

    def can_capture(self) -> bool:
        """Check if captures are allowed in this state."""
        return self == EnumCaptureState.CAPTURING

    def can_transition_to(self, target: "EnumCaptureState") -> bool:
        """
        Check if transition to target state is valid.

        Args:
            target: The target state to transition to.

        Returns:
            True if the transition is valid.
        """
        valid_transitions: dict[EnumCaptureState, set[EnumCaptureState]] = {
            EnumCaptureState.IDLE: {EnumCaptureState.READY},
            EnumCaptureState.READY: {
                EnumCaptureState.CAPTURING,
                EnumCaptureState.CLOSED,
            },
            EnumCaptureState.CAPTURING: {
                EnumCaptureState.PAUSED,
                EnumCaptureState.FULL,
                EnumCaptureState.CLOSED,
            },
            EnumCaptureState.PAUSED: {
                EnumCaptureState.CAPTURING,
                EnumCaptureState.CLOSED,
            },
            EnumCaptureState.FULL: {EnumCaptureState.CLOSED},
            EnumCaptureState.CLOSED: set(),  # Terminal state
        }
        return target in valid_transitions.get(self, set())
