# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Append-only cutover journal event kinds."""

from enum import StrEnum


class EnumCutoverEventKind(StrEnum):
    """Mechanically ordered events in one coherent family cutover."""

    BACKFILL_STARTED = "backfill_started"
    BACKFILL_COMPLETED = "backfill_completed"
    DUAL_WRITE_STARTED = "dual_write_started"
    DUAL_WRITE_ENDED = "dual_write_ended"
    FINAL_DELTA_APPLIED = "final_delta_applied"
    WRITER_CHECKPOINT = "writer_checkpoint"
    APPLICATION_PATH_WRITE_PROVEN = "application_path_write_proven"
    READER_CUTOVER = "reader_cutover"
    OBSERVATION_WINDOW_STARTED = "observation_window_started"
    OBSERVATION_WINDOW_COMPLETED = "observation_window_completed"
    WRITER_QUIESCED = "writer_quiesced"
    REVERSE_DELTA_PROVEN = "reverse_delta_proven"
    FORWARD_FIX_RECORDED = "forward_fix_recorded"
    PRE_CHECKPOINT_ROLLBACK = "pre_checkpoint_rollback"
    MISMATCH_RESOLVED = "mismatch_resolved"


__all__ = ["EnumCutoverEventKind"]
