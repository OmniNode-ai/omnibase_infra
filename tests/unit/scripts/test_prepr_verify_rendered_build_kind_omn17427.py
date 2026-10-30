# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: the rendered-slot verifier admits the image-build kind, and only it.

Measured on 2026-10-30 on the lab host's pre-PR slot 1 at the dev head: all twelve
images built, then ``prepr_verify_lane.sh`` refused to start and exited 12 with
"build-only service dev-lane-infra-runtime-image renders build, command and
entrypoint beyond the slot policy". OMN-21433 had taught the verifier to accept
that service, but its test planted a hand-written shape (``profiles``, ``scale``,
``networks``) that compose never renders. The real render of a service that
exists to be built carries its ``build`` section, and ``docker compose config``
writes an unset ``command`` and ``entrypoint`` as an explicit ``null``.

The policy was the part that was wrong. The compose service cannot move its
build to the image step: the slot's runtime services take it as a ``service:``
build context (OMN-21279), so it has to be a compose service with a build. The
policy therefore gets an entry for the kind, and the entry is exactly the kind's
shape: a ``build`` mapping of ``context``, ``dockerfile`` and ``args``, and
``command`` and ``entrypoint`` only as ``null``. Every other attribute stays
refused, and a service outside the one named build-only service is still refused
as a shared dependency.

The first test renders the real chain, so the next compose release that changes
the rendered shape fails here and not on the lab host.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_BUILD = REPO_ROOT / "scripts" / "runtime_build"
BUILD_ONLY_IMAGE = "dev-lane-infra-runtime-image"
COMPOSE_CHAIN = (
    REPO_ROOT / "docker" / "docker-compose.infra.yml",
    REPO_ROOT / "docker" / "docker-compose.dev-lane.yml",
    REPO_ROOT / "docker" / "docker-compose.prepr.yml",
)


def _load(name: str) -> Any:
    path = RUNTIME_BUILD / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_modules() -> tuple[Any, Any]:
    # Dataclasses need their module registered while it executes. The verifier
    # also prepends its directory to sys.path; neither change may escape here.
    with patch.dict(sys.modules), patch.object(sys, "path", sys.path.copy()):
        return _load("prepr_slot_policy"), _load("prepr_verify_rendered_slot")


policy, verifier = _load_modules()


def test_loading_the_verifier_restores_module_and_path_state() -> None:
    before_modules = sys.modules.copy()
    before_path = sys.path
    _load_modules()
    assert sys.modules == before_modules
    assert sys.path is before_path


@pytest.fixture(scope="module")
def real_build_only_service(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The build-only service exactly as ``docker compose config`` renders it."""
    scratch = tmp_path_factory.mktemp("prepr-build-kind-render")
    runtime_policy = REPO_ROOT / "docker" / "runtime-policy.env"
    policy_keys = set(
        re.findall(r"^([A-Z_][A-Z_0-9]*)=", runtime_policy.read_text(), re.M)
    )
    required = {
        match.group(1)
        for path in (
            *COMPOSE_CHAIN,
            REPO_ROOT / "docker" / "docker-compose.broker-profile.yml",
        )
        for match in re.finditer(r"\$\{([A-Z_][A-Z_0-9]*):?\?", path.read_text())
    }
    project = "prepr-build-kind-render"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", str(scratch)),
        "COMPOSE_PROJECT_NAME": project,
        **dict.fromkeys(required - policy_keys, "render-only"),
        **{key: "48080" for key in required - policy_keys if key.endswith("_PORT")},
        "ONEX_PREPR_SLOT": "1",
        "ONEX_DB_SLOT": "prepr1",
        "PREPR_VALKEY_DB_INDEX": "11",
        "ONEX_LANE_OVERLAY_DIR": str(scratch),
        "ONEX_PREPR_TENANT_STATE_DIR": str(scratch),
        "ONEX_LAB_TENANT_STATE_DIR": str(scratch),
    }
    command = ["docker", "compose", "--env-file", str(runtime_policy), "-p", project]
    for path in COMPOSE_CHAIN:
        command.extend(("-f", str(path)))
    command.extend(("--profile", "prepr", "config", "--format", "json"))
    done = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert done.returncode == 0, done.stderr
    service: dict[str, Any] = json.loads(done.stdout)["services"][BUILD_ONLY_IMAGE]
    return service


def _slot_render() -> dict[str, Any]:
    """Supply the runtime services required alongside a build-only image."""
    slot = policy.resolve_slot(1)
    return {
        "services": {
            name: {
                "container_name": f"omninode-prepr-{slot.slot}-{name}",
                "environment": {
                    "KAFKA_TOPIC_NAMESPACE": slot.topic_namespace,
                    "KAFKA_ENVIRONMENT": slot.kafka_environment,
                    "VALKEY_DB": str(slot.valkey_db_index),
                },
            }
            for name in verifier.SLOT_SERVICES
        }
    }


def _render_with(service: dict[str, Any]) -> list[str]:
    """Verify a good slot render that carries ``service`` as the build-only image."""
    rendered = _slot_render()
    rendered["services"][BUILD_ONLY_IMAGE] = copy.deepcopy(service)
    problems: list[str] = verifier.check(rendered, policy.resolve_slot(1), False)
    return problems


def test_the_real_render_of_the_build_only_image_is_accepted(
    real_build_only_service: dict[str, Any],
) -> None:
    assert "build" in real_build_only_service, "the render lost the build section"
    assert _render_with(real_build_only_service) == []


#: The shape compose rendered on the lab host, written out so the acceptance
#: does not depend on a docker client being present.
RENDERED_SHAPE: dict[str, Any] = {
    "profiles": ["prepr"],
    "build": {
        "context": "/data/omninode/workspace/omnibase_infra",
        "dockerfile": "docker/Dockerfile.runtime",
        "args": {"BUILD_SOURCE": "release", "EXPECTED_BUILD_SOURCE": "release"},
    },
    "command": None,
    "entrypoint": None,
    "networks": {"omnibase-infra-network": None},
    "scale": 0,
}


def test_the_rendered_shape_the_lab_host_refused_is_accepted() -> None:
    assert _render_with(RENDERED_SHAPE) == []


@pytest.mark.parametrize(
    "build_extra",
    [
        {"network": "host"},
        {"ssh": ["default"]},
        {"secrets": ["token"]},
        {"additional_contexts": {"x": "/"}},
        {"target": "debug"},
    ],
)
def test_a_build_section_beyond_context_dockerfile_and_args_is_refused(
    build_extra: dict[str, Any],
) -> None:
    service = copy.deepcopy(RENDERED_SHAPE)
    service["build"].update(build_extra)
    problems = _render_with(service)
    assert any(BUILD_ONLY_IMAGE in p and "build section" in p for p in problems)


def test_a_build_section_that_is_not_a_mapping_is_refused() -> None:
    service = copy.deepcopy(RENDERED_SHAPE)
    service["build"] = "."
    problems = _render_with(service)
    assert any(BUILD_ONLY_IMAGE in p and "build section" in p for p in problems)


@pytest.mark.parametrize("key", ["command", "entrypoint"])
@pytest.mark.parametrize("value", [["sleep", "infinity"], "sleep infinity", ""])
def test_a_command_or_entrypoint_with_a_value_is_refused(
    key: str, value: object
) -> None:
    service = copy.deepcopy(RENDERED_SHAPE)
    service[key] = value
    problems = _render_with(service)
    assert any(BUILD_ONLY_IMAGE in p and key in p for p in problems)


@pytest.mark.parametrize(
    "extra",
    [
        {"image": "busybox"},
        {"ports": ["8085:8085"]},
        {"volumes": ["/:/host"]},
        {"privileged": True},
        {"deploy": {"replicas": 1}},
        {"environment": {"X": "1"}},
    ],
)
def test_every_other_attribute_stays_refused_beside_a_build(
    extra: dict[str, Any],
) -> None:
    assert _render_with(RENDERED_SHAPE) == []
    problems = _render_with({**RENDERED_SHAPE, **extra})
    assert problems == [
        f"build-only service '{BUILD_ONLY_IMAGE}' renders "
        f"attribute(s) {sorted(extra)} beyond the kind's shape."
    ]


def test_scale_other_than_zero_is_refused_beside_a_build() -> None:
    problems = _render_with({**RENDERED_SHAPE, "scale": 1})
    assert any(BUILD_ONLY_IMAGE in p and "scale" in p for p in problems)


def test_the_kind_entry_does_not_admit_a_second_build_service() -> None:
    """The entry is by kind, and the kind has one named member."""
    rendered = _slot_render()
    rendered["services"][BUILD_ONLY_IMAGE] = copy.deepcopy(RENDERED_SHAPE)
    rendered["services"]["another-image"] = copy.deepcopy(RENDERED_SHAPE)
    problems = verifier.check(rendered, policy.resolve_slot(1), False)
    assert any("not the slot's" in p and "another-image" in p for p in problems)
