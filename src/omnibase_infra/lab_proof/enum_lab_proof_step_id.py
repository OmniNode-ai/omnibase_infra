# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Every step a lab proof plan can contain, named so a verdict can find it.

Ticket: OMN-19565
"""

from __future__ import annotations

from enum import StrEnum


class EnumLabProofStepId(StrEnum):
    """Every step a lab proof plan can contain."""

    # --- setup ----------------------------------------------------------------
    HOST_LOAD = "host_load"
    PREFLIGHT_PROJECT_ABSENT = "preflight_project_absent"
    PREFLIGHT_IMAGE_ABSENT = "preflight_image_absent"
    PREFLIGHT_PORTS_FREE = "preflight_ports_free"
    INFRA_INIT = "infra_init"
    INFRA_FETCH = "infra_fetch"
    INFRA_CHECKOUT = "infra_checkout"
    INFRA_REV = "infra_rev"
    SUBJECT_INIT = "subject_init"
    SUBJECT_FETCH = "subject_fetch"
    SUBJECT_CHECKOUT = "subject_checkout"
    SUBJECT_REV = "subject_rev"
    FOCUSED_TESTS = "focused_tests"
    SUBJECT_ARCHIVE = "subject_archive"
    SUBJECT_EXTRACT = "subject_extract"
    NEGATIVE_CONTROL_SABOTAGE = "negative_control_sabotage"
    SUBJECT_HASH = "subject_hash"
    OVERRIDE_DOCKERFILE = "override_dockerfile"
    LOCAL_ENV = "local_env"
    MODEL_ENDPOINT = "model_endpoint"
    GENERATE = "generate"
    BUILD_BASE = "build_base"
    TAG_BASE = "tag_base"
    BUILD_OVERRIDE = "build_override"
    UP = "up"
    # --- prove ----------------------------------------------------------------
    HEALTH_MIGRATION_GATE = "health_migration_gate"
    HEALTH_RUNTIME_MAIN = "health_runtime_main"
    HEALTH_RUNTIME_EFFECTS = "health_runtime_effects"
    IDENTITY_RUNTIME_MAIN = "identity_runtime_main"
    IDENTITY_RUNTIME_EFFECTS = "identity_runtime_effects"
    IMPORT_SMOKE_RUNTIME_MAIN = "import_smoke_runtime_main"
    IMPORT_SMOKE_RUNTIME_EFFECTS = "import_smoke_runtime_effects"
    WIRING_LOGS_RUNTIME_MAIN = "wiring_logs_runtime_main"
    WIRING_LOGS_RUNTIME_EFFECTS = "wiring_logs_runtime_effects"
    GOLDEN_CHAIN_DELEGATION = "golden_chain_delegation"
    # --- teardown -------------------------------------------------------------
    DOWN = "down"
    REMOVE_IMAGES = "remove_images"
    REMOVE_WORKDIR = "remove_workdir"
    # --- residue --------------------------------------------------------------
    RESIDUE_CONTAINERS = "residue_containers"
    RESIDUE_VOLUMES = "residue_volumes"
    RESIDUE_NETWORKS = "residue_networks"
    RESIDUE_IMAGES = "residue_images"
    RESIDUE_PORTS = "residue_ports"
    RESIDUE_WORKDIR = "residue_workdir"
    RESIDUE_POSITIVE_CONTROL = "residue_positive_control"


__all__ = ["EnumLabProofStepId"]
