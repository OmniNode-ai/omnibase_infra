# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15645: static (no-Docker) regression guards for the
``DELEGATION_ROUTING_TIERS_PATH`` binding.

omnimarket#2000 (OMN-15628) removed the packaged-default fallback for this key
in the delegation routing reducer's ``_get_config()`` singleton
(``resolve_required_path_config("DELEGATION_ROUTING_TIERS_PATH")`` —
omnimarket ``src/omnimarket/nodes/node_delegation_routing_reducer/handlers/
handler_delegation_routing.py:392-393``, wrapping
``src/omnimarket/inference/delegation_config_provenance.py:126-158``). An
unbound (or blank) key now raises ``ProtocolConfigurationError`` at first
config read instead of silently defaulting — verified live against merged
``dev`` in the OMN-15645 PR body (a real, non-mocked drive of
``_get_config()``, not this file).

Seam contract with the omnimarket consumer (documented here since omnimarket
is not a pyproject dependency of omnibase_infra and cannot be imported by this
repo's own test suite):

* env-var key name: ``DELEGATION_ROUTING_TIERS_PATH`` (exact string, no alias).
* blank-is-absent: the consumer reads via ``os.environ.get(key, "").strip()``
  (``delegation_config_provenance.py:225``) — an empty string is treated
  identically to unset. The ``""`` opt-out used below for services with no
  delegation-routing surface relies on this exact semantics.
* required, no fallback: ``resolve_required_path_config`` raises ``ValueError``
  (wrapped into ``ProtocolConfigurationError`` by the handler) when the
  resolved value is falsy — there is no packaged/bootstrap default for this
  key (unlike ``BIFROST_CONTRACT_PATH``, which uses the optional
  ``resolve_path_config`` variant and is NOT in scope for this ticket).

These tests are deliberately static (text/regex over the Dockerfile and
compose file content, no ``docker`` invocation) so they fire on hosts without
Docker. The real, end-to-end proof (real container, real omnimarket wheel,
real routing-reducer invocation, sha256-verified packaged file) is the
``docker compose config`` integration tests in
``tests/integration/infra/test_{dev,stability_test,judge,prod}*compose_render*.py``
plus the GREEN cold bring-up captured in the OMN-15645 PR body.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

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
def test_dockerfile_bakes_the_packaged_file_at_the_expected_fixed_path(
    dockerfile_content: str,
) -> None:
    """Dockerfile.runtime must COPY the installed omnimarket routing_tiers.yaml
    to the SAME fixed path the compose anchor declares — the two loci must
    never drift independently."""
    assert _EXPECTED_PATH in dockerfile_content, (
        f"docker/Dockerfile.runtime must bake a file at {_EXPECTED_PATH!r} "
        "(the same fixed path docker-compose.infra.yml's x-runtime-env anchor "
        f"binds {_ENV_KEY} to)."
    )
    assert "omnimarket/configs/routing_tiers.yaml" in dockerfile_content, (
        "docker/Dockerfile.runtime must source the baked file from the "
        "installed omnimarket package's own configs/routing_tiers.yaml."
    )


@pytest.mark.unit
def test_dockerfile_copy_source_uses_a_glob_never_a_literal_python_version(
    dockerfile_content: str,
) -> None:
    """The Dockerfile COPY --from=builder source must glob the interpreter
    minor version (``python*``), matching the existing omnibase_core runtime
    contracts COPY precedent, never hardcode e.g. ``python3.12``."""
    for line in dockerfile_content.splitlines():
        if "omnimarket/configs/routing_tiers.yaml" in line and "COPY" not in line:
            # Continuation line of a multi-line COPY; still must not embed a
            # literal version.
            assert not _VERSION_EMBEDDED_LITERAL.search(line), (
                f"Dockerfile COPY source line embeds a literal python3.X "
                f"site-packages path: {line!r}. Use the python* glob instead "
                "(matches the existing omnibase_core runtime contracts COPY)."
            )
    assert (
        "/app/.venv/lib/python*/site-packages/omnimarket/configs/routing_tiers.yaml"
        in (dockerfile_content)
    ), (
        "docker/Dockerfile.runtime must glob the venv's python* directory when "
        "copying the packaged routing_tiers.yaml, never hardcode a python3.X "
        "literal (the exact base-image-Python-bump trap OMN-15628's entrypoint "
        "self-heal exists to correct for a *stale* pin)."
    )


@pytest.mark.unit
def test_dockerfile_bake_runs_after_the_venv_copy(dockerfile_content: str) -> None:
    """The routing_tiers.yaml bake must run AFTER the builder venv (which
    contains the installed omnimarket package) is copied into the image."""
    venv_copy_idx = dockerfile_content.find(
        "COPY --from=builder --chown=omniinfra:omniinfra /app/.venv /app/.venv"
    )
    bake_idx = dockerfile_content.find(_EXPECTED_PATH)
    assert venv_copy_idx != -1, "venv COPY step missing from Dockerfile.runtime"
    assert bake_idx != -1, (
        "routing_tiers.yaml bake step missing from Dockerfile.runtime"
    )
    assert venv_copy_idx < bake_idx, (
        "the routing_tiers.yaml bake COPY must run AFTER the venv COPY step "
        "(the source file only exists in the image once the venv has landed)"
    )


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
