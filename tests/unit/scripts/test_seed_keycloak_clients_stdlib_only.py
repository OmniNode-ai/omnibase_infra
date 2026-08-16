# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deployment-seam pins for the Keycloak realm reconciler (OMN-16026).

``scripts/seed-keycloak-clients.py`` is invoked as a batch Job -- omninode_infra's
``k8s/onex-dev/jobs/seed-keycloak-clients-job.yaml`` runs
``python scripts/seed-keycloak-clients.py`` with ``workingDir: /app`` against the
digest-pinned ``omninode-runtime`` image.

Two properties have to hold for that to work, and both were violated in
production simultaneously on 2026-08-13:

1. The file must be present in the runtime image at ``/app/scripts/``. It was
   not -- the ``runtime`` stage of ``docker/Dockerfile.runtime`` had no ``COPY``
   of ``scripts/`` at all, so every Job run died with
   ``python: can't open file '/app/scripts/seed-keycloak-clients.py'``. The
   earlier fix attempt (omnibase_infra#2661) was closed without merging.
2. The module must stay import-cheap. It is run as a plain file, not via
   ``python -m``, precisely so it does not pay the ``omnibase_infra`` package
   import cost -- the same reasoning as ``onex-container-healthcheck``.

Both are pinned here rather than left as comments, because a comment did not
stop (1) from regressing for months.

Related Tickets:
    - OMN-16026: seed-keycloak-clients Job cannot run -- script absent from image
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_SCRIPT = REPO_ROOT / "scripts" / "seed-keycloak-clients.py"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.runtime"

#: The path the omninode_infra Job's ``command``/``workingDir`` pair resolves to.
INVOKED_PATH = "/app/scripts/seed-keycloak-clients.py"


@pytest.mark.unit
class TestSeedKeycloakClientsDeploymentSeam:
    """Pins that keep the reconciler runnable where it is actually invoked."""

    def test_script_exists(self) -> None:
        assert SEED_SCRIPT.is_file(), f"missing reconciler at {SEED_SCRIPT}"

    def test_module_imports_only_the_standard_library(self) -> None:
        """Kept stdlib-only so the Job can run it as a plain file.

        The Dockerfile copies this to a bare path instead of relying on
        ``python -m omnibase_infra...`` for the documented import-cost reason.
        A single first-party import would silently invalidate that choice.
        """
        tree = ast.parse(SEED_SCRIPT.read_text(encoding="utf-8"))

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])

        non_stdlib = {
            name
            for name in imported
            if name not in sys.stdlib_module_names and name != "__future__"
        }
        assert non_stdlib == set(), (
            "seed-keycloak-clients.py must stay stdlib-only (it is run as a "
            f"plain file by the onex-dev Job); found {sorted(non_stdlib)}"
        )

    def test_dockerfile_installs_the_script_at_the_invoked_path(self) -> None:
        """The regression guard for OMN-16026.

        Without this COPY the Job fails at exec time, not build time, so
        nothing in CI notices -- which is exactly how it stayed broken.
        """
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        assert INVOKED_PATH in dockerfile, (
            f"{DOCKERFILE.name} must COPY the reconciler to {INVOKED_PATH}; the "
            "onex-dev seed Job execs that literal path against this image."
        )
