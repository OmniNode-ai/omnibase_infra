# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18073: every credential the compose surface carries must be a declared mapping.

The dev lane runs with ``enable_convention_fallback: false``. Under that
posture a logical name that is not an explicit ``secret_resolver_mappings``
entry resolves to nothing, and the resolver records the miss as reason
``no_mapping`` -- logged as "Secret resolution failed (configuration issue)"
and otherwise invisible, because every call site threads
``env_var_fallback=ref`` and quietly reads ``os.environ`` instead.

That silent fallback is exactly what hid the OMN-18073 transport defect: the
resolver never claimed ownership of ``ONEXBOT_OCC_APP_ID`` /
``ONEXBOT_OCC_PRIVATE_KEY``, so when the value in the container env became
unusable there was no failure on the store path for anyone to act on. It is
the same defect the OMN-16921 ``SLACK_BOT_TOKEN`` entry was added to close.

This test generalises the fix rather than pinning the two names: the compose
allowlist is the set of credentials a lane's containers can receive, so every
name on it that a node resolves by literal ref must be a declared mapping.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT = _REPO_ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"

# Credentials resolved by LITERAL ref (contract_secret_ref returns the node
# contract's own secrets-block dict key, not a dotted convention name), so a
# convention fallback would not rescue them even if it were enabled.
_LITERAL_REF_CREDENTIALS = (
    "ONEXBOT_OCC_APP_ID",
    "ONEXBOT_OCC_PRIVATE_KEY",
)

# The lane whose runtime mints OCC companions.
_LANE = "dev"


def _lane_mappings(lane: str) -> list[dict[str, object]]:
    doc = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    profiles = doc["profiles"]
    assert lane in profiles, f"lane {lane!r} absent from {_CONTRACT}"
    mappings = profiles[lane].get("secret_resolver_mappings")
    assert mappings, f"lane {lane!r} declares no secret_resolver_mappings"
    return list(mappings)


def _logical_names(lane: str) -> set[str]:
    return {str(m["logical_name"]) for m in _lane_mappings(lane)}


@pytest.mark.parametrize("credential", _LITERAL_REF_CREDENTIALS)
def test_literal_ref_credential_is_a_declared_mapping(credential: str) -> None:
    """RED before the fix: neither OCC name was a mapped logical_name."""
    names = _logical_names(_LANE)
    assert credential in names, (
        f"{credential} is carried by the {_LANE} lane's compose environment but is "
        f"not a declared secret_resolver mapping. Under this lane's "
        f"enable_convention_fallback: false posture the resolver will record "
        f"reason=no_mapping and the call site's env_var_fallback will read "
        f"os.environ silently instead -- the OMN-18073 defect."
    )


@pytest.mark.parametrize("credential", _LITERAL_REF_CREDENTIALS)
def test_declared_mapping_names_a_source(credential: str) -> None:
    """A declared mapping must carry a resolvable source, not just a name."""
    by_name = {str(m["logical_name"]): m for m in _lane_mappings(_LANE)}
    source = by_name[credential]["source"]
    assert isinstance(source, dict)
    assert source.get("source_type") in {"env", "file", "infisical"}
    assert str(source.get("source_path", "")).strip(), (
        f"{credential} declares source_type {source.get('source_type')!r} with an "
        "empty source_path, which resolves to nothing at runtime."
    )


def test_convention_fallback_stays_off_in_the_renderer() -> None:
    """The premise of this test file: an undeclared name really does miss.

    ``render_runtime_policy_env`` builds every lane's resolver config with
    ``enable_convention_fallback=False``. If that ever flipped, the assertions
    above would still pass while no longer proving anything, so the posture
    they depend on is pinned here rather than assumed.
    """
    from omnibase_infra.runtime.models.model_secret_resolver_config import (
        ModelSecretResolverConfig,
    )

    renderer = (_REPO_ROOT / "scripts" / "render_runtime_policy_env.py").read_text(
        encoding="utf-8"
    )
    assert "enable_convention_fallback=False" in renderer

    # The renderer is the ONLY thing pinning this off: the model's own default
    # is permissive. Recorded rather than asserted-away, because it means a
    # future lane built without the renderer would silently get the fallback
    # back, and these assertions would then pass while proving nothing.
    assert ModelSecretResolverConfig(mappings=[]).enable_convention_fallback is True
