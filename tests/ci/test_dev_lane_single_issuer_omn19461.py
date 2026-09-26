# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The .201 dev lane has one token issuer, and every validator expects it (OMN-19461).

Before this, no token could satisfy both halves of a gateway attach on the lane.
Keycloak stamped ``iss`` from the Host of each token request, the runtime
services validated the browser-visible issuer (``KEYCLOAK_ISSUER``), and
onex-api validated the in-network one (``KEYCLOAK_ISSUER_URL``), because it also
used that value to find its signing keys. Every ``ga-*`` attach returned 502 on
``TokenValidationError`` at the node, and OMN-19373 AC1 could not be run.

Three properties hold the fix together. If any one of them drifts, the lane
still boots and still reports healthy while gateway attach is dead again:

* Keycloak pins its frontend URL, so ``iss`` does not depend on who asked, and
  keeps its backchannel dynamic, so containers still reach it in-network.
* onex-api expects exactly the issuer the runtime services expect.
* onex-api fetches its keys in-network, never from the browser-visible host,
  which from inside a container is the container itself.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_LANE = REPO_ROOT / "docker" / "docker-compose.dev-lane.yml"
BASE = REPO_ROOT / "docker" / "docker-compose.infra.yml"


def _load(path: Path) -> dict:
    text = path.read_text().replace("!!merge ", "").replace("!override", "")
    return yaml.safe_load(text)


def _runtime_issuer_default() -> str:
    """The value the runtime services validate, from the base x-runtime-env block."""
    raw = _load(BASE)["x-runtime-env"]["KEYCLOAK_ISSUER"]
    # ${KEYCLOAK_ISSUER:-<default>}
    assert raw.startswith("${KEYCLOAK_ISSUER:-") and raw.endswith("}"), raw
    return raw[len("${KEYCLOAK_ISSUER:-") : -1]


def _onex_api_env() -> dict:
    return _load(DEV_LANE)["services"]["onex-api"]["environment"]


def _keycloak_env() -> dict:
    return _load(DEV_LANE)["services"]["keycloak"]["environment"]


@pytest.mark.unit
def test_onex_api_expects_the_issuer_the_runtime_validates() -> None:
    assert _onex_api_env()["KEYCLOAK_ISSUER_URL"] == _runtime_issuer_default()


@pytest.mark.unit
def test_onex_api_fetches_keys_in_network() -> None:
    env = _onex_api_env()
    key_source = urlsplit(env["KEYCLOAK_JWKS_URL"])
    issuer = urlsplit(env["KEYCLOAK_ISSUER_URL"])
    admin = urlsplit(env["KEYCLOAK_ADMIN_BASE_URL"])

    assert key_source.netloc == admin.netloc == "keycloak:8080"
    assert key_source.netloc != issuer.netloc
    assert env["KEYCLOAK_JWKS_URL"] == (
        f"http://keycloak:8080{issuer.path}/protocol/openid-connect/certs"
    )


@pytest.mark.unit
def test_keycloak_stamps_the_issuer_whoever_asks() -> None:
    env = _keycloak_env()
    frontend = urlsplit(env["KC_HOSTNAME"])
    issuer = urlsplit(_runtime_issuer_default())

    assert (frontend.scheme, frontend.netloc) == (issuer.scheme, issuer.netloc)
    # Without a dynamic backchannel every container would be sent to
    # localhost:28080 for tokens, introspection and keys, which is itself.
    assert str(env["KC_HOSTNAME_BACKCHANNEL_DYNAMIC"]).lower() == "true"


@pytest.mark.unit
def test_the_hostname_pin_is_scoped_to_the_dev_lane() -> None:
    """Other lanes publish their own Keycloak on another port (stability-test:
    38080); a base value would name this lane's port for them."""
    base_env = _load(BASE)["services"]["keycloak"]["environment"]
    assert "KC_HOSTNAME" not in base_env
    assert "KC_HOSTNAME_BACKCHANNEL_DYNAMIC" not in base_env
