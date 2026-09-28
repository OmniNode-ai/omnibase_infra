# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Why a provenance verdict was reached (OMN-19927)."""

from __future__ import annotations

from enum import StrEnum


class EnumMergeProvenanceReason(StrEnum):
    """One reason per way the verdict can be reached."""

    # VALIDATED: a merge-group run of the workflow for this exact sha has a
    # summary job that concluded success.
    MERGE_GROUP_SUMMARY_SUCCESS = "merge_group_summary_success"
    # UNVALIDATED: the read succeeded and found no merge-group run for the sha
    # (a direct merge, a direct push, or a commit the queue never built).
    NO_MERGE_GROUP_RUN = "no_merge_group_run"
    # UNVALIDATED: merge-group runs exist, and none has a successful summary
    # (failed, cancelled, or still running).
    NO_SUCCESSFUL_SUMMARY = "no_successful_summary"
    # UNDECIDABLE: the read failed or was incomplete.
    READ_FAILED = "read_failed"
    # UNDECIDABLE: runs exist and at least one has no job of the summary name,
    # so the lookup key itself may be stale.
    SUMMARY_JOB_ABSENT = "summary_job_absent"
    # UNDECIDABLE: the observation describes another repository or sha.
    OBSERVATION_MISMATCH = "observation_mismatch"


__all__: list[str] = ["EnumMergeProvenanceReason"]
