# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The compose model declares no migration-gate service (OMN-24483).

The long-running ``migration-gate`` container (a database image running a
keepalive that runtime services waited on) is replaced by the runtime's own
boot wait. The falsifier: this test fails while any compose file or catalog
manifest still declares the service or still depends on it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO_ROOT / "docker"

#: A service key (``  migration-gate:``), a compose dependency key under
#: ``depends_on`` (``      migration-gate:``) or a catalog dependency entry
#: (``  - service: migration-gate``). Comments are stripped before matching.
_DECLARES_OR_DEPENDS = re.compile(
    r"^\s*(?:-\s+service:\s*)?migration-gate\s*:?\s*(?:\*\w+)?\s*$"
)


def _hits(text: str) -> list[str]:
    hits = []
    for line in text.splitlines():
        code = line.split("#", 1)[0]
        if _DECLARES_OR_DEPENDS.match(code):
            hits.append(line.strip())
    return hits


def _model_files() -> list[Path]:
    compose = sorted(DOCKER_DIR.glob("docker-compose*.yml"))
    catalog = sorted((DOCKER_DIR / "catalog").rglob("*.yaml"))
    return [*compose, *catalog]


def test_scan_sees_a_declaration_and_a_dependency() -> None:
    """Positive control: the scan fails on the shape the model used to carry."""
    declared = "services:\n  migration-gate:\n    image: postgres:16-alpine\n"
    depended = (
        "    depends_on:\n      migration-gate:\n        condition: service_healthy\n"
    )
    catalogued = (
        "depends_on:\n  - service: migration-gate\n    condition: service_healthy\n"
    )
    aliased = "  migration-gate: *prepr_fenced\n"
    for text in (declared, depended, catalogued, aliased):
        assert _hits(text), text
    assert not _hits("  # migration-gate was retired (OMN-24483)\n")


def test_model_files_were_found() -> None:
    files = _model_files()
    assert any(f.name == "docker-compose.infra.yml" for f in files)
    assert any(f.name == "docker-compose.dev-lane.yml" for f in files)
    assert any(f.parent.name == "services" for f in files)


def test_no_compose_or_catalog_file_declares_or_depends_on_migration_gate() -> None:
    offenders = {
        str(path.relative_to(REPO_ROOT)): _hits(path.read_text(encoding="utf-8"))
        for path in _model_files()
    }
    offenders = {name: hits for name, hits in offenders.items() if hits}
    assert not offenders, offenders


def test_no_catalog_manifest_is_named_migration_gate() -> None:
    assert not (DOCKER_DIR / "catalog" / "services" / "migration-gate.yaml").exists()


def test_the_dev_lane_model_waits_on_forward_migration_instead() -> None:
    text = (DOCKER_DIR / "docker-compose.dev-lane.yml").read_text(encoding="utf-8")
    assert re.search(
        r"forward-migration:\n\s+condition: service_completed_successfully", text
    )


def test_runtime_services_carry_the_boot_wait_opt_in() -> None:
    infra = (DOCKER_DIR / "docker-compose.infra.yml").read_text(encoding="utf-8")
    assert 'ONEX_BOOT_WAIT_FOR_MIGRATIONS: "true"' in infra
    for name in ("omninode-runtime", "runtime-effects", "runtime-worker"):
        manifest = (DOCKER_DIR / "catalog" / "services" / f"{name}.yaml").read_text(
            encoding="utf-8"
        )
        assert "ONEX_BOOT_WAIT_FOR_MIGRATIONS: 'true'" in manifest, name
