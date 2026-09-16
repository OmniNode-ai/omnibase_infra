# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pins the local-LLM HMAC key wiring for the runner fleet (OMN-18415).

The Hostile Review Gate's reviewer (omniintelligence's ``cli_review``) reaches
the .201 inference endpoint through omnibase_infra's ``MixinLlmHttpTransport``,
which HMAC-signs every request with ``LOCAL_LLM_SHARED_SECRET`` and fails
CLOSED per model when it is absent. Before this ticket the key was on no runner
in the ``omnibase-ci`` fleet, so every model call failed; a separate shell
defect (OMN-18409) then laundered that hard failure into a non-blocking
``degraded`` verdict, so the gate reported reviews it had never performed.

Coverage:
- Every one of the 60 ``omninode-runner-N`` fleet services -- config-resolved
  through the real YAML merge keys, never regexed -- carries the key.
- The interpolation is FAIL-CLOSED (``:?``), not a soft default. An empty
  value still satisfies the transport's "is it set" check and yields a wrong
  signature, i.e. a 401 with no named cause: the same silent class this ticket
  removes.
- ``omninode-deploy-runner`` and both ``omninode-customer-plane-runner-N``
  services are out of scope and must NOT receive it -- the customer-plane pair
  is credential-free by construction (OMN-18392).
- No secret VALUE is committed: the compose file carries an interpolation
  expression only. The value lives in ``docker/.env`` on the runner host and is
  absent from ``deploy-runners.sh``'s ``SYNC_PATHS``, so a repo-side rsync can
  neither publish nor blank it -- the same rule the lab credentials directory
  follows.

Every zero-result assertion here is paired with a positive control, because a
membership test over a parsed mapping passes vacuously if the parse silently
yields the wrong shape.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.runners.yml"
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"

SECRET_ENV_VAR = "LOCAL_LLM_SHARED_SECRET"
# The reviewer's call path fails closed on BOTH of these before any HTTP is
# attempted: the signing key, and the trust boundary the transport refuses to
# default for itself. Provisioning one without the other moves the failure
# three seconds later in the same call path and changes nothing else.
CIDR_ENV_VAR = "LLM_ENDPOINT_CIDR_ALLOWLIST"
REQUIRED_REVIEWER_ENV_VARS = (SECRET_ENV_VAR, CIDR_ENV_VAR)
# OMN-18411: fleet capped 88 -> 60.
FLEET_SERVICE_COUNT = 60
OUT_OF_SCOPE_SERVICES = (
    "omninode-deploy-runner",
    "omninode-customer-plane-runner-1",
    "omninode-customer-plane-runner-2",
)
# A variable the fleet deliberately does NOT set: the reviewer's model registry
# carries a live ``default_url`` for this endpoint, so CI needs no override.
# Used as the positive control for every "is present" assertion below.
ABSENT_CONTROL_ENV_VAR = "LLM_DEEPSEEK_R1_URL"


def _load_compose() -> dict[str, Any]:
    loaded = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _fleet_services(compose: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fleet = {
        name: svc
        for name, svc in compose["services"].items()
        if name.startswith("omninode-runner-")
    }
    assert len(fleet) == FLEET_SERVICE_COUNT, (
        f"expected {FLEET_SERVICE_COUNT} omninode-runner-N services, found "
        f"{len(fleet)}: {sorted(fleet)}"
    )
    return fleet


def _env(service: dict[str, Any]) -> dict[str, Any]:
    env = service.get("environment", {})
    assert isinstance(env, dict), (
        "a list-form `environment:` would defeat the merge key this wiring "
        f"relies on: {env!r}"
    )
    return env


# --- every fleet service resolves the key -----------------------------------


def test_every_fleet_service_carries_the_local_llm_shared_secret() -> None:
    compose = _load_compose()
    for var in REQUIRED_REVIEWER_ENV_VARS:
        missing = [
            name
            for name, svc in _fleet_services(compose).items()
            if var not in _env(svc)
        ]
        assert not missing, f"fleet services missing {var}: {missing}"


def test_the_presence_assertion_can_actually_fail() -> None:
    """Positive control for the test above.

    A membership check over a mapping that parsed into an unexpected shape
    would pass for every key. This proves the same check reports a variable the
    fleet genuinely does not set, so a green result above means the key is
    really there.
    """
    compose = _load_compose()
    absent = [
        name
        for name, svc in _fleet_services(compose).items()
        if ABSENT_CONTROL_ENV_VAR not in _env(svc)
    ]
    assert len(absent) == FLEET_SERVICE_COUNT, (
        f"{ABSENT_CONTROL_ENV_VAR} was expected on no fleet service; the "
        f"control is no longer valid: {sorted(set(_fleet_services(compose)) - set(absent))}"
    )


def test_the_key_reaches_services_through_the_shared_anchor() -> None:
    """One declaration, not sixty. ``environment:`` is a mapping, so
    ``!!merge <<: *runner-env`` extends it -- unlike ``volumes:``, a list,
    which each service must duplicate (the OMN-16363/OMN-18188 precedent).
    Sixty hand-copied declarations would pass the test above while being the
    exact drift hazard the anchor exists to prevent.
    """
    raw = COMPOSE_FILE.read_text(encoding="utf-8")
    for var in REQUIRED_REVIEWER_ENV_VARS:
        declarations = re.findall(rf"^\s*{var}:", raw, flags=re.MULTILINE)
        assert len(declarations) == 1, (
            f"expected exactly one {var} declaration (on the &runner-env "
            f"anchor), found {len(declarations)}"
        )


# --- the interpolation is fail-closed ---------------------------------------


def test_the_interpolation_is_fail_closed_not_a_soft_default() -> None:
    compose = _load_compose()
    for var in REQUIRED_REVIEWER_ENV_VARS:
        for name, svc in _fleet_services(compose).items():
            value = _env(svc)[var]
            assert isinstance(value, str)
            assert value.startswith(f"${{{var}:?"), (
                f"{name}: {var} must use the fail-fast `:?` guard, not a "
                f"`:-` default or a literal -- got {value[:40]!r}..."
            )


def test_no_secret_value_is_committed_to_the_repository() -> None:
    """The compose file may name the variable; it may never carry its value."""
    raw = COMPOSE_FILE.read_text(encoding="utf-8")
    assert f"{SECRET_ENV_VAR}=" not in raw, (
        f"{COMPOSE_FILE.name} contains a literal {SECRET_ENV_VAR} assignment"
    )
    compose = _load_compose()
    for var in REQUIRED_REVIEWER_ENV_VARS:
        for name, svc in _fleet_services(compose).items():
            value = _env(svc)[var]
            assert value.startswith("${") and value.endswith("}"), (
                f"{name}: {var} must be an interpolation expression, never an "
                "inline value"
            )


# --- scope boundary ---------------------------------------------------------


def test_out_of_scope_runners_do_not_receive_the_key() -> None:
    """The deploy runner and the customer-plane pair define their own
    ``environment:`` blocks and do not merge ``&runner-env``. The
    customer-plane pair is credential-free by construction (OMN-18392); that
    property must not regress through this anchor.
    """
    compose = _load_compose()
    leaked = [
        (name, var)
        for name in OUT_OF_SCOPE_SERVICES
        for var in REQUIRED_REVIEWER_ENV_VARS
        if var in _env(compose["services"][name])
    ]
    assert not leaked, f"reviewer env leaked to out-of-scope services: {leaked}"


def test_the_scope_boundary_control_services_exist() -> None:
    """Positive control for the test above: a typo in a service name would
    make the leak check pass by testing nothing.
    """
    compose = _load_compose()
    for name in OUT_OF_SCOPE_SERVICES:
        assert name in compose["services"], f"unknown service in scope list: {name}"
        assert name not in _fleet_services(compose)


# --- the host value is never rsynced from this repo -------------------------


def test_the_host_env_file_is_not_rsynced_from_this_repo() -> None:
    """``docker/.env`` on the runner host holds the value. An rsync entry would
    let a repo-side deploy overwrite or blank a host-generated secret, exactly
    as the lab credentials directory must not be synced.
    """
    script_text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    start = script_text.index("SYNC_PATHS=(")
    end = script_text.index(")", start)
    sync_block = script_text[start:end]
    for forbidden in ("docker/.env", '".env"', "docker/env"):
        assert forbidden not in sync_block, (
            f"SYNC_PATHS must not carry {forbidden!r}: the runner host's env "
            "file is a host-generated secret, not a build artifact"
        )
    # Positive control: the block really is the sync list and really does carry
    # the compose file itself, so the absence above is a finding, not an empty
    # slice.
    assert "docker/docker-compose.runners.yml" in sync_block


def test_the_required_env_var_is_documented_where_operators_look() -> None:
    """A `:?`-guarded var that nobody documents is a fleet that fails to come
    up with no pointer to the fix.
    """
    script_text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    start = script_text.index("CONSOLIDATED REQUIRED ENV VARS")
    end = script_text.index("set -euo pipefail")
    block = script_text[start:end]
    assert SECRET_ENV_VAR in block
    assert "docker/.env" in block, (
        "the documentation must name where the value is read from, or the "
        "fail-closed guard is undiagnosable"
    )
