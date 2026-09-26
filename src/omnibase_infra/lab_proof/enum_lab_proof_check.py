# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The check names a profile may declare as mandatory for a PASS.

A registry row names its mandatory checks from this set, and the verdict handler
evaluates exactly the checks a row names. A name outside this set is refused by
the registry validator, so a misspelled check cannot silently stop being
required.

Ticket: OMN-19565
"""

from __future__ import annotations

from enum import StrEnum


class EnumLabProofCheck(StrEnum):
    """The check names a profile may declare as mandatory for a PASS."""

    SUBJECT_HEAD_IDENTITY = "subject_head_identity"
    FOCUSED_TESTS = "focused_tests"
    OVERRIDE_INSTALLED_IDENTITY = "override_installed_identity"
    MIGRATION_GATE_HEALTHY = "migration_gate_healthy"
    RUNTIME_MAIN_HEALTHY = "runtime_main_healthy"
    RUNTIME_EFFECTS_HEALTHY = "runtime_effects_healthy"
    CONSUMER_IMPORT_SMOKE = "consumer_import_smoke"
    NO_WIRING_FAILURES = "no_wiring_failures"
    GOLDEN_CHAIN_DELEGATION = "golden_chain_delegation"
    # Hand-run recipe checks (profiles with execution manual_recipe). Declared so
    # a hand-run row names its checks in the same vocabulary a node row does.
    RUNTIME_IMAGE_IDENTITY = "runtime_image_identity"
    CHANGED_PATH_LIVE = "changed_path_live"
    SCRIPT_REPLAY_READ_ONLY = "script_replay_read_only"
    RENDER_AND_BROWSER_WALK = "render_and_browser_walk"


__all__ = ["EnumLabProofCheck"]
