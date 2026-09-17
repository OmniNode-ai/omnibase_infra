# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18478 AC3: the compose generator refuses to render a redaction placeholder.

``generate_compose`` merges ``hardcoded_env`` into a service's rendered
``environment`` verbatim (``generator.py``), and nothing downstream can tell the
secret redactor's replacement token apart from a real value. A container started
from such a render carries a DSN that cannot authenticate, and the failure
surfaces at first query rather than at render time.

The guard fails the render instead.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver

CATALOG_DIR = str(Path(__file__).resolve().parents[3] / "docker" / "catalog")

# Built from parts rather than spelled -- see the module docstring of
# tests/test_no_redaction_placeholder_in_docker.py.
REDACTION_PLACEHOLDER = "*" * 3 + "REDACTED" + "*" * 3


def _resolved_core_stack() -> object:
    return CatalogResolver(catalog_dir=CATALOG_DIR).resolve(bundles=["core"])


@pytest.mark.unit
def test_generate_compose_rejects_placeholder_in_hardcoded_env() -> None:
    resolved = _resolved_core_stack()
    name = "postgres"
    manifest = resolved.manifests[name]
    poisoned = dict(manifest.hardcoded_env)
    poisoned["OMNIBASE_INFRA_DB_URL"] = (
        f"postgresql://postgres:{REDACTION_PLACEHOLDER}@postgres:5432/omnibase_infra"
    )
    resolved.manifests[name] = replace(manifest, hardcoded_env=poisoned)

    with pytest.raises(ValueError) as excinfo:
        generate_compose(resolved)

    message = str(excinfo.value)
    assert name in message
    assert "OMNIBASE_INFRA_DB_URL" in message


@pytest.mark.unit
def test_generate_compose_renders_the_clean_catalog() -> None:
    """Negative control: the guard does not fire on the tree as it stands.

    A guard that raised unconditionally would pass the test above while breaking
    every render, so the passing case is asserted in the same file.
    """
    compose = generate_compose(_resolved_core_stack())
    assert compose["services"]
