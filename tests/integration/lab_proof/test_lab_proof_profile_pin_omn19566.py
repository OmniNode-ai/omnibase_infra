# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``profile_pin`` resolved against the committed lab-proof registry (OMN-19566).

A publisher of a pr-head receipt pins the profile key, version and mandatory
checks from the reviewed registry rather than from its own caller (see
``ModelLabProofProfilePin``). This proves that wiring end to end against the
real, committed ``config/lab_proof_profiles.yaml`` -- not a unit-test stub --
since a mismatch between the registry's shape and ``profile_pin``'s reader is
exactly the class of bug a mocked registry cannot catch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from omnibase_infra.lab_proof.enum_lab_proof_check import EnumLabProofCheck
from omnibase_infra.lab_proof.enum_lab_proof_execution import EnumLabProofExecution
from omnibase_infra.lab_proof.enum_lab_proof_kind import EnumLabProofKind
from omnibase_infra.lab_proof.enum_lab_proof_profile_status import (
    EnumLabProofProfileStatus,
)
from omnibase_infra.lab_proof.lab_proof_profile_registry import (
    load_lab_proof_profile_registry,
    profile_pin,
)
from omnibase_infra.lab_proof.model_lab_proof_profile_pin import (
    ModelLabProofProfilePin,
)
from omnibase_infra.lab_proof.model_lab_proof_profile_registry import (
    ModelLabProofProfileRegistry,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = REPO_ROOT / "config" / "lab_proof_profiles.yaml"

INFRA_REPO = "OmniNode-ai/omnibase_infra"
CORE_REPO = "OmniNode-ai/omnibase_core"


def _registry() -> ModelLabProofProfileRegistry:
    loaded = load_lab_proof_profile_registry(REGISTRY)
    return ModelLabProofProfileRegistry.model_validate(loaded.model_dump())


def test_pin_resolves_the_runtime_image_variant_from_the_committed_registry() -> None:
    registry = _registry()

    pin = profile_pin(registry, INFRA_REPO, EnumLabProofKind.RUNTIME_IMAGE)

    assert isinstance(pin, ModelLabProofProfilePin)
    assert pin.repo == INFRA_REPO
    assert pin.proof_kind is EnumLabProofKind.RUNTIME_IMAGE
    assert pin.profile_key == "omnibase_infra.runtime_and_scripts"
    assert pin.profile_version.isdigit()
    assert len(pin.mandatory_checks) > 0
    assert all(isinstance(check, EnumLabProofCheck) for check in pin.mandatory_checks)
    # OMN-19566's finding for the profile: a migration-only PR has no runtime
    # log line, so `changed_path_live` must stay pinned as mandatory here --
    # weakening that pin is exactly what `profile_pin` exists to prevent.
    assert EnumLabProofCheck.CHANGED_PATH_LIVE in pin.mandatory_checks


def test_pin_resolves_a_second_proof_kind_on_the_same_repo_row() -> None:
    registry = _registry()

    pin = profile_pin(registry, INFRA_REPO, EnumLabProofKind.SCRIPT_REPLAY)

    assert pin.proof_kind is EnumLabProofKind.SCRIPT_REPLAY
    assert pin.profile_key == "omnibase_infra.runtime_and_scripts"
    assert len(pin.mandatory_checks) > 0


def test_pin_is_frozen_and_rejects_unknown_fields() -> None:
    registry = _registry()
    pin = profile_pin(registry, INFRA_REPO, EnumLabProofKind.RUNTIME_IMAGE)

    with pytest.raises(Exception):
        pin.repo = "OmniNode-ai/other"  # type: ignore[misc]


def test_pin_raises_key_error_for_a_repo_the_registry_does_not_carry() -> None:
    registry = _registry()

    with pytest.raises(KeyError):
        profile_pin(registry, "OmniNode-ai/does-not-exist", EnumLabProofKind.EXEMPT)


def test_pin_raises_value_error_when_the_repo_has_no_variant_of_that_kind() -> None:
    registry = _registry()

    # omnibase_core's committed row is foundation_override only.
    with pytest.raises(ValueError, match="has no"):
        profile_pin(registry, CORE_REPO, EnumLabProofKind.RUNTIME_IMAGE)


def test_pin_raises_value_error_when_the_matching_variant_names_no_check(
    tmp_path: Path,
) -> None:
    raw = cast("dict[str, Any]", yaml.safe_load(REGISTRY.read_text(encoding="utf-8")))
    profiles = cast("list[dict[str, Any]]", raw["profiles"])
    infra = next(p for p in profiles if p["repo"] == INFRA_REPO)

    # A NOT_SAFE variant may legally carry an empty `mandatory_checks`: it
    # never runs, so nothing enforces its non-emptiness at the model layer.
    # `profile_pin` must refuse to hand out a pin with no check to enforce
    # rather than silently accepting any PASS.
    stub_variant = {
        "variant_key": "stub_not_safe",
        "proof_kind": EnumLabProofKind.WEB_RENDER.value,
        "status": EnumLabProofProfileStatus.NOT_SAFE.value,
        "status_reason": "OMN-19566 test stub: never runs, names no check",
        "execution": EnumLabProofExecution.NONE.value,
    }
    infra["variants"] = [*infra["variants"], stub_variant]

    stub_registry_path = tmp_path / "lab_proof_profiles.yaml"
    stub_registry_path.write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )

    loaded = load_lab_proof_profile_registry(stub_registry_path)
    registry = ModelLabProofProfileRegistry.model_validate(loaded.model_dump())

    with pytest.raises(ValueError, match="names no mandatory check"):
        profile_pin(registry, INFRA_REPO, EnumLabProofKind.WEB_RENDER)
