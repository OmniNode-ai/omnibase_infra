# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Local bring-up must not require credentials only a private vault can supply.

OMN-16187. A clean-room self-install proof found that ``onex up core`` — the
documented local bring-up — refused to start until eight env vars were set,
three of which (``INFISICAL_CLIENT_ID``, ``INFISICAL_CLIENT_SECRET``,
``INFISICAL_PROJECT_ID``) an operator cannot possess before the first bring-up:
they are *minted* by ``scripts/setup-infisical-identity.sh`` at step 4 of
``bootstrap-infisical.sh``, against the Infisical container that ``onex up core``
(steps 1-3) is what starts. They were also consumed by nothing in the core
stack — bundle ``inject_required_env`` is a pure validation gate that the
generator never writes into any container's environment (only per-service
``required_env`` reaches a container).

These tests pin the two properties that keep the local tier self-hostable:

1. Every var ``core`` requires is one the operator can generate locally, and
2. ``.env.example`` names all of them, so the template is never again a subset
   of what the bring-up enforces.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from omnibase_infra.docker.catalog.resolver import CatalogResolver

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_CATALOG_DIR = _PROJECT_ROOT / "docker" / "catalog"
_BUNDLES_PATH = _CATALOG_DIR / "bundles.yaml"
_ENV_EXAMPLE = _PROJECT_ROOT / ".env.example"
_GENERATOR = _PROJECT_ROOT / "scripts" / "generate-local-env.sh"

# Credentials for an Infisical *machine identity*. These are an OUTPUT of
# bootstrap (step 4), never an input to it, so no bundle that a first-time
# operator brings up may require them.
_MACHINE_IDENTITY_VARS = frozenset(
    {
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
        "INFISICAL_PROJECT_ID",
    }
)

# Bundles a self-hosting operator brings up before any identity exists.
_FIRST_BRINGUP_BUNDLES = ("core", "canary", "omnidash", "omnimarket-projections")


@pytest.fixture(scope="module")
def bundles() -> dict[str, Any]:
    with open(_BUNDLES_PATH) as f:
        return cast("dict[str, Any]", yaml.safe_load(f))


def _resolved_required_env(bundle: str) -> set[str]:
    resolver = CatalogResolver(catalog_dir=str(_CATALOG_DIR))
    return resolver.resolve(bundles=[bundle]).required_env


@pytest.mark.unit
@pytest.mark.parametrize("bundle", _FIRST_BRINGUP_BUNDLES)
def test_first_bringup_bundles_do_not_require_machine_identity(bundle: str) -> None:
    """A bundle brought up before bootstrap step 4 cannot require step 4's output."""
    offending = _resolved_required_env(bundle) & _MACHINE_IDENTITY_VARS
    assert not offending, (
        f"Bundle {bundle!r} requires Infisical machine-identity credentials "
        f"{sorted(offending)}, which are minted by "
        "scripts/setup-infisical-identity.sh only AFTER the Infisical container "
        "this bundle starts is already running. Requiring them here is a "
        "chicken-and-egg gate that makes local bring-up impossible for a "
        "self-hosting operator (OMN-16187)."
    )


@pytest.mark.unit
def test_runtime_bundle_still_requires_machine_identity(
    bundles: dict[str, Any],
) -> None:
    """The gate is moved to the runtime lane, not deleted.

    ``runtime_host_process`` is the only consumer; keeping the requirement there
    preserves the operator contract that the runtime never starts with config
    prefetch silently disabled.
    """
    runtime = bundles["runtime"]
    assert isinstance(runtime, dict)
    required = set(runtime.get("inject_required_env", []))
    assert _MACHINE_IDENTITY_VARS.issubset(required), (
        "runtime bundle must keep requiring the Infisical machine identity so "
        "config prefetch cannot be silently skipped in the runtime lane."
    )


def _env_example_assigned_vars() -> set[str]:
    """Var names with an uncommented assignment in ``.env.example``."""
    assigned: set[str] = set()
    for raw_line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Z_][A-Z0-9_]*)=", line)
        if match:
            assigned.add(match.group(1))
    return assigned


@pytest.mark.unit
def test_env_example_covers_every_core_required_var() -> None:
    """The front-door template must not be a subset of what bring-up enforces.

    This is the ratchet: adding a ``required_env`` entry to any core service now
    fails here until ``.env.example`` documents it, so the clean-room dead end
    cannot silently reappear.
    """
    missing = _resolved_required_env("core") - _env_example_assigned_vars()
    assert not missing, (
        f".env.example is missing required core vars {sorted(missing)}. Every var "
        "'onex up core' enforces must appear as an uncommented assignment in the "
        "template a new operator copies (OMN-16187)."
    )


@pytest.mark.unit
def test_local_env_generator_emits_every_core_required_var() -> None:
    """``generate-local-env.sh`` must produce a complete, working local secret set."""
    assert _GENERATOR.exists(), f"missing local env generator at {_GENERATOR}"
    body = _GENERATOR.read_text(encoding="utf-8")
    emitted = {m.group(1) for m in re.finditer(r"^([A-Z_][A-Z0-9_]*)=", body, re.M)}
    missing = _resolved_required_env("core") - emitted
    assert not missing, (
        f"generate-local-env.sh does not emit {sorted(missing)}; running it must "
        "leave 'onex up core' able to start with no further manual editing."
    )
