# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15645: static guards for the explicit routing-tiers deployment pin.

The shared compose anchor and runtime image already bind and bake
``DELEGATION_ROUTING_TIERS_PATH=/app/config/delegation/routing_tiers.yaml``.
Catalog renders must carry the same pin on main, effects and worker kernels.
These tests exercise the catalog resolver and generator without Docker and
check that the pinned image path has no Python version or shadowing mount.

The ticket originally described a consumer that refused an unbound key.
The current lab worker resolves an unset or blank key to packaged tiers, so
these checks assert the explicit deployment binding rather than that historical
refusal. Services without a delegation-routing surface retain their blank
opt-outs. Container loader probes and lane bring-up evidence are separate from
these static checks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver

# The exact key name the omnimarket consumer reads. Do not rename without
# updating the omnimarket-side seam citation above.
_ENV_KEY = "DELEGATION_ROUTING_TIERS_PATH"

# The fixed, non-version-embedded in-image path this PR binds the key to.
# Single source of truth for the value asserted across every render fixture.
_EXPECTED_PATH = "/app/config/delegation/routing_tiers.yaml"

# A literal python3.<minor> site-packages path is exactly the trap OMN-15628's
# runtime entrypoint self-heal exists to correct for a *stale* pin — the
# compose default itself must never regress into one.
_VERSION_EMBEDDED_LITERAL = re.compile(r"python3\.\d+/site-packages")


@pytest.fixture
def recorded_response(
    request: pytest.FixtureRequest, project_root: Path
) -> dict[str, object]:
    """Read the worker contact recorded for the marked catalog test."""
    marker = request.node.get_closest_marker("live_contact")
    assert marker is not None and len(marker.args) == 1
    document = json.loads((project_root / marker.args[0]).read_text())
    assert isinstance(document, dict)
    return document


@pytest.mark.unit
def test_compose_anchor_binds_the_key_to_the_expected_fixed_path(
    compose_file_content: str,
) -> None:
    """x-runtime-env must hardcode DELEGATION_ROUTING_TIERS_PATH (OMN-12864
    shape: no ``${VAR:-}`` ambient-override footgun, same as
    BIFROST_CONTRACT_PATH)."""
    assert f"{_ENV_KEY}: {_EXPECTED_PATH}" in compose_file_content, (
        f"docker-compose.infra.yml's x-runtime-env anchor must bind {_ENV_KEY} "
        f"to the literal {_EXPECTED_PATH!r} (hardcoded default, no ${{VAR:-}} "
        "form)."
    )


@pytest.mark.unit
def test_compose_default_is_never_a_version_embedded_literal(
    compose_file_content: str,
) -> None:
    """The compose-declared default must never regress into a python3.X
    site-packages literal."""
    for line in compose_file_content.splitlines():
        if line.strip().startswith(f"{_ENV_KEY}:"):
            assert not _VERSION_EMBEDDED_LITERAL.search(line), (
                f"docker-compose.infra.yml binds {_ENV_KEY} to a "
                f"version-embedded python3.X site-packages literal: {line!r}. "
                "Use a stable, build-time-baked, non-version-embedded path "
                "instead (see docker/Dockerfile.runtime)."
            )


@pytest.mark.unit
def test_dockerfile_bakes_the_tracked_file_at_the_expected_fixed_path(
    dockerfile_content: str, project_root: Path
) -> None:
    """Dockerfile.runtime must COPY the repo-tracked routing_tiers.yaml to the
    SAME fixed path the compose anchor declares — the two loci must never
    drift independently. The source is a tracked file, never a path inside an
    installed omnimarket wheel (OMN-21279: the image carries no omnimarket)."""
    copy = re.compile(
        r"^COPY\s+(?:--chown=\S+\s+)?(?P<src>\S+)\s+"
        + re.escape(_EXPECTED_PATH)
        + r"\s*$",
        re.MULTILINE,
    ).search(dockerfile_content)
    assert copy is not None, (
        f"docker/Dockerfile.runtime must bake a file at {_EXPECTED_PATH!r} "
        "(the same fixed path docker-compose.infra.yml's x-runtime-env anchor "
        f"binds {_ENV_KEY} to)."
    )
    source = copy.group("src")
    assert not _VERSION_EMBEDDED_LITERAL.search(source)
    assert "site-packages" not in source
    assert (project_root / source).is_file(), (
        f"the COPY source {source!r} is not a tracked file of this repo"
    )


@pytest.mark.unit
def test_workspace_staging_overwrites_the_placeholder_ladder(
    project_root: Path,
) -> None:
    """The committed placeholder is replaced by the staged ladder (OMN-21279).

    The image bakes ``workspace/routing_tiers.yaml``. A placeholder keeps the COPY
    valid for release builds; stage_workspace.sh must overwrite it from the staged
    sibling in workspace mode, and the placeholder must be an empty ladder so a
    placement that names a tier fails loudly rather than reading stale data.
    """
    stage = (
        project_root / "scripts" / "runtime_build" / "stage_workspace.sh"
    ).read_text()
    assert 'ROUTING_TIERS_OUT="workspace/routing_tiers.yaml"' in stage
    assert "shutil.copyfile(sys.argv[1], sys.argv[2])" in stage
    placeholder = yaml.safe_load(
        (project_root / "workspace" / "routing_tiers.yaml").read_text()
    )
    assert placeholder == {"tiers": []}


@pytest.mark.unit
def test_projection_api_and_contract_resolver_opt_out_explicitly(
    compose_file_content: str,
) -> None:
    """Services with no delegation-routing surface must explicitly bind the
    key to "" (relies on the blank-is-absent seam semantics documented above),
    mirroring the existing BIFROST_CONTRACT_PATH opt-out for the same two
    services."""
    assert f'{_ENV_KEY}: ""' in compose_file_content, (
        f'Expected at least one explicit {_ENV_KEY}: "" opt-out in '
        "docker-compose.infra.yml (projection-api / contract-resolver)."
    )
    # Count actual YAML key-value lines only — excludes the prose comment
    # above the anchor binding that also mentions the literal '<KEY>: ""'
    # shape.
    opt_out_count = sum(
        1
        for line in compose_file_content.splitlines()
        if line.strip() == f'{_ENV_KEY}: ""'
    )
    assert opt_out_count == 2, (
        f'Expected exactly 2 explicit {_ENV_KEY}: "" opt-outs '
        "(projection-api, omninode-contract-resolver); found "
        f"{opt_out_count}. If a service's delegation-routing surface changed, "
        "update this count and the render-fixture assertions in "
        "tests/integration/infra/test_*compose_render*.py together."
    )


@pytest.mark.unit
def test_expected_path_is_never_shadowed_by_a_volume_mount(
    compose_file_path: Path,
) -> None:
    """CodeRabbit catch (PR #2620): the bound path must never fall under a
    service's mounted volume target. The runtime services bind-mount
    ``../contracts:/app/contracts:ro`` (host content) and
    ``${OMNICLAUDE_SKILLS_DIR:-./skills}:/app/skills:ro`` — a baked file under
    either of those container-side prefixes would be silently HIDDEN by the
    mount at container start, making the image bake pointless and leaving the
    entrypoint's OMN-15628 self-heal to paper over it. This test parses every
    service's ``volumes:`` list and asserts the expected path's directory is
    never a descendant of any mounted target, for every runtime-profile
    service that inherits the anchor.
    """
    data = yaml.safe_load(compose_file_path.read_text())
    services = data.get("services", {})
    expected_dir = _EXPECTED_PATH.rsplit("/", 1)[0] + "/"

    runtime_services = ("omninode-runtime", "runtime-effects", "runtime-worker")
    violations: list[str] = []
    for service_name in runtime_services:
        service = services.get(service_name)
        if service is None:
            continue
        for volume_entry in service.get("volumes", []) or []:
            if not isinstance(volume_entry, str):
                continue
            # "src:dst[:mode]" — container-side target is the second field.
            parts = volume_entry.split(":")
            if len(parts) < 2:
                continue
            target = parts[1]
            target_dir = target if target.endswith("/") else target + "/"
            if expected_dir.startswith(target_dir):
                violations.append(
                    f"{service_name}: mount {volume_entry!r} shadows {_EXPECTED_PATH!r}"
                )

    assert not violations, (
        "DELEGATION_ROUTING_TIERS_PATH's expected path is shadowed by a "
        "volume mount on at least one runtime service:\n"
        + "\n".join(f"  - {v}" for v in violations)
        + f"\n\nChoose a fixed path outside every mounted target for "
        f"{runtime_services}."
    )


@pytest.mark.unit
@pytest.mark.live_contact("tests/fixtures/omn15645_worker_tiers_live_contact.json")
@pytest.mark.parametrize(
    ("bundle", "service_name"),
    [
        ("runtime-core", "omninode-runtime"),
        ("runtime-core", "runtime-effects"),
        ("runtime", "omninode-runtime"),
        ("runtime", "runtime-effects"),
        ("local", "omninode-runtime"),
        ("local", "runtime-effects"),
        ("runtime", "runtime-worker"),
        ("runtime-infrastructure", "runtime-worker"),
    ],
)
def test_catalog_runtime_services_bind_the_packaged_tiers_path(
    project_root: Path,
    bundle: str,
    service_name: str,
    recorded_response: dict[str, object],
) -> None:
    """Catalog-generated kernels need the same pin as the shared compose anchor."""
    assert recorded_response["runtime_profile"] == "workers"
    assert recorded_response["loader_type"] == "ModelDelegationConfig"
    assert recorded_response["file_sha256"] == recorded_response["packaged_sha256"]
    assert recorded_response["tiers_path"] == _EXPECTED_PATH
    resolver = CatalogResolver(catalog_dir=str(project_root / "docker" / "catalog"))
    compose = generate_compose(resolver.resolve(bundles=[bundle]))
    environment = compose["services"][service_name]["environment"]
    assert environment.get(_ENV_KEY) == recorded_response["tiers_path"], (
        f"{bundle}/{service_name} must bind {_ENV_KEY} to {_EXPECTED_PATH}; "
        f"got {environment.get(_ENV_KEY)!r}"
    )
