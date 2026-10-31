# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev-lane Infisical advertises a SITE_URL on the port it actually listens on.

WHY THIS EXISTS (OMN-20999, AC2). An invite e-mail carries SITE_URL as its link.
The dev lane publishes Infisical on host port 8881 (OMN-13417); the
other lane on the same host is a separate instance with its own database on
8880. The live dev container was found advertising port 8880, so even once mail
worked, an invite link would have opened the *other* instance.

Three places tell an operator what SITE_URL to use, and all three must agree with
the dev lane's published port and stay overridable, so a host can point the link
at an address an operator Mac can reach (the dev host sets its tailnet name):

* the compose service reads ``${INFISICAL_SITE_URL:-...}`` with an 8881 default;
* the catalog manifest's SITE_URL is override-backed the same way, not a literal
  that silently discards ``INFISICAL_SITE_URL``;
* the env example documents 8881, not the other lane's 8880 that the bad
  live value was copied from.

The catalog generator copies ``operational_defaults`` into the rendered service
environment verbatim, and compose substitutes ``${VAR:-default}`` there, so
``test_rendered_core_bundle_passes_the_override_through`` reads the value the
generator actually emits rather than trusting the manifest text.

The compose file carries tags ``yaml.safe_load`` refuses, so it is read as text,
the same way ``test_infisical_smtp_env_omn20999.py`` reads it.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSE = _REPO_ROOT / "docker" / "docker-compose.infra.yml"
_CATALOG = _REPO_ROOT / "docker" / "catalog" / "services" / "infisical.yaml"
_ENV_EXAMPLE = _REPO_ROOT / "docker" / "env-example-full.txt"

_DEV_LANE_PORT = 8881
_OVERRIDE = re.compile(r"^\$\{INFISICAL_SITE_URL:-(https?://[^}]+)\}$")


def _port_of(url: str) -> int:
    port = urlsplit(url).port
    assert port is not None, f"SITE_URL default {url!r} names no port"
    return port


def _compose_site_url() -> str:
    """SITE_URL from the ``environment:`` block of the infisical service only."""
    lines = _COMPOSE.read_text(encoding="utf-8").splitlines()
    start = lines.index("  infisical:")
    in_environment = False
    for line in lines[start + 1 :]:
        if re.match(r"^  [a-z]", line):
            break
        if line.lstrip().startswith("#"):
            continue
        if re.match(r"^    [a-z_]+:", line):
            in_environment = line.strip() == "environment:"
            continue
        found = re.match(r"^      SITE_URL:\s*(.+?)\s*$", line)
        if in_environment and found:
            return found.group(1).strip("'\"")
    raise AssertionError("infisical service environment passes no SITE_URL")


def test_compose_site_url_is_override_backed_on_the_dev_lane_port() -> None:
    value = _compose_site_url()
    match = _OVERRIDE.match(value)
    assert match, f"compose SITE_URL {value!r} is not ${{INFISICAL_SITE_URL:-...}}"
    assert _port_of(match.group(1)) == _DEV_LANE_PORT


def test_catalog_site_url_is_override_backed_on_its_published_port() -> None:
    manifest = yaml.safe_load(_CATALOG.read_text(encoding="utf-8"))
    value = str(manifest["operational_defaults"]["SITE_URL"])
    match = _OVERRIDE.match(value)
    assert match, (
        f"catalog SITE_URL {value!r} is a literal, so INFISICAL_SITE_URL is ignored "
        "and an invite link can never point off-box"
    )
    published = manifest.get("ports", {}).get("external")
    assert published == _DEV_LANE_PORT, f"catalog publishes infisical on {published!r}"
    default_port = _port_of(match.group(1))
    assert default_port == published, (
        f"catalog SITE_URL defaults to port {default_port}, published port is {published}"
    )


def test_env_example_documents_the_dev_lane_port() -> None:
    documented = re.findall(
        r"^#?\s*INFISICAL_SITE_URL=(\S+)$",
        _ENV_EXAMPLE.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert documented, "env example no longer documents INFISICAL_SITE_URL"
    wrong = [url for url in documented if _port_of(url) != _DEV_LANE_PORT]
    assert not wrong, f"env example points SITE_URL at another lane: {wrong}"


def test_rendered_core_bundle_passes_the_override_through() -> None:
    resolved = CatalogResolver(catalog_dir=str(_CATALOG.parents[1])).resolve(["core"])
    services = generate_compose(resolved)["services"]
    assert isinstance(services, dict)
    environment = services["infisical"]["environment"]
    match = _OVERRIDE.match(environment["SITE_URL"])
    assert match, f"rendered SITE_URL {environment['SITE_URL']!r} drops the override"
    assert _port_of(match.group(1)) == _DEV_LANE_PORT
