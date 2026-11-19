# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24562: the stability lane builds the runtime on omnimarket's composed image.

Built from the infra Dockerfile alone, the stability kernel fails boot with
``BIFROST_BASE_CONTRACT_PATH is unset or blank``. The dev lane has built on the
composed image since OMN-21279; the stability overlay mirrors that recipe.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from tests.unit.docker.test_dev_lane_composed_runtime_image_omn21279 import (
    _DOCKER,
    _env,
    dev_stack_services,
    find_services_without_market_image,
    load_compose,
)

pytestmark = pytest.mark.unit

_OVERLAY_FILES = sorted(_DOCKER.glob("docker-compose.stability-t*.yml"))
assert len(_OVERLAY_FILES) == 1, _OVERLAY_FILES
_OVERLAY = _OVERLAY_FILES[0]
_STABILITY_BUILD_ONLY = "composed-runtime-base-image"


def stability_stack_services() -> dict[str, dict[str, Any]]:
    """The stability stack: the base infra services with the stability overlay on."""
    merged = load_compose(_DOCKER / "docker-compose.infra.yml")
    for name, overlay in load_compose(_OVERLAY).items():
        base = dict(merged.get(name) or {})
        env = {**_env(base), **_env(overlay)}
        base.update(overlay)
        if env:
            base["environment"] = env
        merged[name] = base
    return merged


def _composed(services: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: service["build"]
        for name, service in services.items()
        if isinstance(service.get("build"), Mapping)
        and str(service["build"].get("context", "")).endswith(
            "sibling-repos/omnimarket"
        )
    }


def test_stability_overlay_services_build_on_the_composed_image() -> None:
    composed = _composed(load_compose(_OVERLAY))
    assert {
        "omninode-runtime",
        "runtime-effects",
        "runtime-worker",
        "projection-api",
        "tenant-projection-writer",
        "projection-delegation-writer",
        "projection-live-events-writer",
    } <= set(composed)
    for name, build in composed.items():
        assert build["dockerfile"] == "Dockerfile.runtime", name
        assert build["args"]["INFRA_RUNTIME_IMAGE"] == "infra-runtime", name
        assert build["additional_contexts"]["infra-runtime"] == (
            f"service:{_STABILITY_BUILD_ONLY}"
        ), name


def test_stability_infra_half_is_build_only_and_lane_named() -> None:
    services = stability_stack_services()
    infra = services[_STABILITY_BUILD_ONLY]
    assert infra["build"]["context"] == ".."
    assert infra["build"]["dockerfile"] == "docker/Dockerfile.runtime"
    assert infra["scale"] == 0
    assert "runtime" in infra["profiles"]
    assert "container_name" not in infra


def _runtime_profile_services() -> dict[str, dict[str, Any]]:
    """Services a runtime bring-up starts: the opt-in and disabled ones never do."""
    return {
        name: service
        for name, service in stability_stack_services().items()
        if {"runtime", "full"} & set(service.get("profiles") or [])
    }


def test_real_stability_stack_has_no_market_workloads_on_the_infra_only_image() -> None:
    assert _runtime_profile_services()["omninode-runtime"]
    assert (
        find_services_without_market_image(
            _runtime_profile_services(),
            (_DOCKER / "Dockerfile.runtime").read_text(encoding="utf-8"),
        )
        == {}
    )


def test_control_the_infra_only_stability_build_is_refused() -> None:
    """Positive control: the pre-fix shape (infra Dockerfile, market command) is flagged."""
    services = stability_stack_services()
    services["omninode-runtime"] = {
        **services["omninode-runtime"],
        "build": {"context": "..", "dockerfile": "docker/Dockerfile.runtime"},
        "environment": {"BIFROST_CONTRACT_PATH": "/app/data/bifrost.yaml"},
    }
    assert "omninode-runtime" in find_services_without_market_image(services, "")


def test_dev_composed_build_set_is_unchanged_by_the_stability_overlay() -> None:
    dev = _composed(dev_stack_services())
    assert dev, "no composed runtime services found"
    for build in dev.values():
        assert build["additional_contexts"]["infra-runtime"] == (
            "service:dev-lane-infra-runtime-image"
        )
    assert _STABILITY_BUILD_ONLY not in dev_stack_services()
