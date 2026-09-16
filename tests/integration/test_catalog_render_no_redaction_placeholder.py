# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18478: the rendered compose carries no placeholder and no spelled password.

This exercises the path the defect actually reached: the catalog CLI render that
``onex up <bundle>`` starts services from
(``src/omnibase_infra/docker/catalog/cli.py``). The unit tests assert the source
manifests and the generator guard; this asserts the artifact a container is
actually built from, through the real CLI entrypoint rather than an in-process
call, so a regression in argument handling or bundle resolution is caught too.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Built from parts, never spelled -- a literal here is rewritten in any Bash
# preview of this file. See tests/test_no_redaction_placeholder_in_docker.py.
REDACTION_PLACEHOLDER = "*" * 3 + "REDACTED" + "*" * 3

_DSN = re.compile(r"[a-z+]+://([^:@\s]+):([^@\s]+)@")
_EXPANSION = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*(?:(?::?[-?+])[^}]*)?\}$")


def _render(bundles: list[str], output: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnibase_infra.docker.catalog.cli",
            "generate",
            *bundles,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, f"render failed: {result.stderr}"


@pytest.mark.integration
def test_rendered_compose_binds_no_placeholder_or_spelled_password(
    tmp_path: Path,
) -> None:
    output = tmp_path / "docker-compose.generated.yml"
    _render(["core", "runtime"], output)

    compose = yaml.safe_load(output.read_text(encoding="utf-8"))
    services = compose["services"]
    assert services, "render produced no services"

    offenders: list[str] = []
    for name, svc in services.items():
        for key, value in (svc.get("environment") or {}).items():
            for match in _DSN.finditer(str(value)):
                segment = match.group(2)
                if REDACTION_PLACEHOLDER in segment:
                    offenders.append(f"{name}.{key} PLACEHOLDER")
                elif not _EXPANSION.match(segment):
                    offenders.append(f"{name}.{key} LITERAL")

    assert offenders == [], (
        "The rendered compose binds a credential that is not resolved from the "
        f"environment: {offenders}. A container started from this render would "
        "carry a DSN that cannot authenticate."
    )


@pytest.mark.integration
def test_render_refuses_a_manifest_carrying_the_placeholder(tmp_path: Path) -> None:
    """Positive control: the render fails rather than emitting the token.

    Without this, the assertion above would pass just as happily against a render
    step that had silently stopped emitting DSNs at all.
    """
    catalog = tmp_path / "catalog"
    subprocess.run(
        ["cp", "-R", str(REPO_ROOT / "docker" / "catalog"), str(catalog)], check=True
    )
    poisoned = catalog / "services" / "postgres.yaml"
    lines = poisoned.read_text(encoding="utf-8").splitlines()
    index = lines.index("hardcoded_env:")
    lines.insert(
        index + 1,
        f"  OMNIBASE_INFRA_DB_URL: 'postgresql://postgres:{REDACTION_PLACEHOLDER}@postgres:5432/d'",
    )
    poisoned.write_text("\n".join(lines) + "\n", encoding="utf-8")

    from omnibase_infra.docker.catalog.generator import generate_compose
    from omnibase_infra.docker.catalog.resolver import CatalogResolver

    resolved = CatalogResolver(catalog_dir=str(catalog)).resolve(bundles=["core"])
    with pytest.raises(ValueError, match="redactor"):
        generate_compose(resolved)
