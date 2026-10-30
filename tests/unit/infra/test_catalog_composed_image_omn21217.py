# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The composed laptop stack: every runtime-image service on one image (OMN-21217).

omnibase_infra's runtime image installs no omnimarket (OMN-21279). The lab proof
of an omnimarket change therefore has to boot the stack on the composed image
omnimarket builds from this repo's image, and it has to know which containers
run that image so it can read the omnimarket package inside each of them. These
tests pin both: the render puts the composed image on every service that runs
the runtime image or the market image and drops the infra build stanza, and the
catalog CLI names those services' containers from the same render, so no list of
container names is kept anywhere else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.docker.catalog import cli as catalog_cli
from omnibase_infra.docker.catalog.generator import (
    containers_on_image,
    generate_compose,
)
from omnibase_infra.docker.catalog.resolver import CatalogResolver

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_CATALOG_DIR = str(_REPO / "docker" / "catalog")
_PROJECT = "omnibase-infra-local"
_COMPOSED = f"{_PROJECT}-omnimarket-runtime:latest"
_INFRA_IMAGE = f"{_PROJECT}-runtime:latest"


def _services(compose: dict[str, object]) -> dict[str, dict[str, object]]:
    services = compose["services"]
    assert isinstance(services, dict)
    return services


def _render(bundle: str, *, composed_image: str | None = None) -> dict[str, object]:
    resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve([bundle])
    return generate_compose(resolved, composed_image=composed_image)


def test_the_render_without_a_composed_image_keeps_the_kernels_on_the_infra_image() -> (
    None
):
    services = _services(_render("local-market"))
    for name in ("omninode-runtime", "runtime-effects"):
        assert services[name]["image"] == _INFRA_IMAGE
    assert "build" in services["omninode-runtime"]


def test_a_composed_image_replaces_the_runtime_image_on_every_service_that_runs_it() -> (
    None
):
    services = _services(_render("local-market", composed_image=_COMPOSED))
    on_image = {n for n, s in services.items() if s["image"] == _COMPOSED}
    assert {
        "omninode-runtime",
        "runtime-effects",
        "projection-api",
        "omnimarket-projection-delegation",
        "omnimarket-projection-tenant-credentials",
        "omnimarket-projection-llm-cost",
        "omnimarket-projection-run-trace",
    } <= on_image
    # nothing on the infra image is left over, and no service builds it
    assert all(s["image"] != _INFRA_IMAGE for s in services.values())
    assert all("build" not in services[n] for n in on_image)
    # a service on neither image keeps its own
    assert services["postgres"]["image"] != _COMPOSED


def test_containers_on_image_reads_the_names_from_the_render() -> None:
    compose = _render("local-market", composed_image=_COMPOSED)
    names = containers_on_image(compose, _COMPOSED)
    assert f"{_PROJECT}-omninode-runtime" in names
    assert f"{_PROJECT}-runtime-effects" in names
    assert f"{_PROJECT}-omnimarket-projection-delegation" in names
    assert f"{_PROJECT}-postgres" not in names
    services = _services(compose)
    assert names == sorted(
        str(s["container_name"]) for s in services.values() if s["image"] == _COMPOSED
    )


def test_an_image_no_service_runs_names_no_container() -> None:
    compose = _render("local-market", composed_image=_COMPOSED)
    assert containers_on_image(compose, "unrelated:latest") == []


def test_the_cli_prints_one_container_per_line_for_the_image(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = catalog_cli.cmd_image_containers([_COMPOSED, "local-market"])
    assert rc == 0
    printed = capsys.readouterr().out.split()
    assert f"{_PROJECT}-omninode-runtime" in printed
    assert f"{_PROJECT}-projection-api" in printed
    assert f"{_PROJECT}-postgres" not in printed


def test_the_cli_refuses_a_call_that_names_no_bundle(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = catalog_cli.cmd_image_containers([_COMPOSED])
    captured = capsys.readouterr()
    assert rc != 0
    assert captured.out.strip() == ""
    assert "image-containers" in captured.err
