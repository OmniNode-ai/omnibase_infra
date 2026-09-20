# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay (OMN-15547) for the OMN-14631 workspace content-parity gate.

THE INCIDENT, and it is a FALSE RED rather than a false green. On 2026-09-19
at 15:37Z every ``BUILD_SOURCE=workspace`` image build began dying with::

    INSTALLED CONTENT DRIFT for 'omnibase-core' (OMN-14631): ...
    differing files (1 total): ['data/gitignore-baseline.yaml']

and the .201 dev lane could not rebuild at all. Nothing was wrong with the
tree. omnibase_core#1710 (squash ``5d562ab4``) had added a hatch
``force-include`` so the governance spec ships inside the wheel for
downstream pre-commit environments::

    [tool.hatch.build.targets.wheel.force-include]
    "architecture-handshakes/gitignore-baseline.yaml" =
        "omnibase_core/data/gitignore-baseline.yaml"

A force-included file is in the installed tree by construction and can NEVER
be in the staged ``src/<pkg>`` tree, so the gate read it as an extra
installed file -- as drift -- and refused a wheel that was correct.
omnibase_infra#3846 (OMN-18847) taught the gate to RESOLVE the declared
mapping instead of tolerating extra installed files.

WHY THE VERDICT ON THE ARTIFACT IS ``accept``. The buggy gate got this
direction wrong: it said FAIL on a real GOOD input. The replay therefore
requires the real resolver to resolve the real declaration, out of the real
captured ``pyproject.toml``.

WHY THE DISCRIMINATOR IS MANDATORY HERE, more than usual. An accept-only
proof cannot tell a working resolver from a DISABLED one: a resolver that
returned "nothing is drift, ever" would pass the accept case perfectly and
silently retire the entire OMN-14631 proof, which exists because a 2026-07-14
rebuild shipped stale installed config while the staged source was correct.
The discriminator drives the same real function over the same real bytes with
the declared source REMOVED, and requires a recorded error: a declared
mapping whose source is absent means the staged tree is not the tree the
wheel was built from, which is the condition the gate exists to refuse.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "runtime_build" / "compute_workspace_provenance.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn18847"

PYPROJECT = FIXTURES / "omnibase_core-1710.pyproject.toml.captured"
PYPROJECT_SHA256 = "cde44beaeebe785a8a8c228bebde55fefe99a3c875b6fa9046b0fbaaff453e4e"
SPEC = FIXTURES / "omnibase_core-1710.gitignore-baseline.yaml.captured"
SPEC_SHA256 = "f617d04432a45e6bb448200af5e9ef5b3d29e92c98a08363bfc3dfd59ea16371"

# The exact destination the image gate named in its refusal, and the source
# path the captured pyproject maps it from.
DECLARED_SOURCE = "architecture-handshakes/gitignore-baseline.yaml"
DECLARED_DESTINATION = "data/gitignore-baseline.yaml"


def _load_gate() -> object:
    sys.path.insert(0, str(GATE.parent))
    spec = importlib.util.spec_from_file_location("workspace_gate_for_replay", GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage(root: Path, *, with_source: bool) -> Path:
    """Reproduce the staged sibling tree the image build hands the gate.

    ``stage_workspace.sh`` rsyncs the repo without ``.git``, so the staged
    tree is the working tree minus transient artefacts -- which is what this
    lays down, with the captured bytes verbatim.
    """
    (root / "pyproject.toml").write_bytes(PYPROJECT.read_bytes())
    pkg = root / "src" / "omnibase_core"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    if with_source:
        source = root / DECLARED_SOURCE
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(SPEC.read_bytes())
    return root


def test_the_captured_artifacts_are_the_bytes_they_claim_to_be() -> None:
    """A fixture whose digest has moved is no longer the artifact that failed."""
    assert hashlib.sha256(PYPROJECT.read_bytes()).hexdigest() == PYPROJECT_SHA256
    assert hashlib.sha256(SPEC.read_bytes()).hexdigest() == SPEC_SHA256
    assert "[tool.hatch.build.targets.wheel.force-include]" in PYPROJECT.read_text()


def test_the_real_gate_resolves_the_real_declaration(tmp_path: Path) -> None:
    """ACCEPT: the gate must resolve the mapping that took the lane down.

    Before omnibase_infra#3846 this file's declaration produced
    ``differing files (1 total): ['data/gitignore-baseline.yaml']`` on every
    workspace build. The resolver must now return that destination carrying
    the SOURCE's real bytes, with no error recorded.
    """
    gate = _load_gate()
    root = _stage(tmp_path, with_source=True)
    errors: list[str] = []

    resolved = gate._force_included_files(root, "omnibase_core", errors)  # type: ignore[attr-defined]

    assert errors == [], errors
    assert DECLARED_DESTINATION in resolved, (
        "the gate did not resolve the force-include that broke every "
        f"workspace build at 15:37Z; it returned {sorted(resolved)}"
    )
    # Byte-for-byte, not merely present: resolving the mapping is a narrowing
    # of the comparison, never an exemption from it.
    assert resolved[DECLARED_DESTINATION] == SPEC.read_bytes()


def test_a_declared_source_missing_from_the_staged_tree_is_recorded_as_an_error(
    tmp_path: Path,
) -> None:
    """The discriminator: resolution must not become a blanket tolerance.

    A resolver that answered "nothing is drift" would pass the accept case
    above and quietly retire the whole OMN-14631 proof. With the declared
    source absent, the staged tree is not the tree the wheel was built from,
    and the gate must say so rather than read it as "nothing declared".
    """
    gate = _load_gate()
    root = _stage(tmp_path, with_source=False)
    errors: list[str] = []

    resolved = gate._force_included_files(root, "omnibase_core", errors)  # type: ignore[attr-defined]

    assert resolved == {}
    assert errors, "a declared mapping with no source must fail closed"
    assert DECLARED_SOURCE in errors[0]
    assert "not the tree the wheel was built from" in errors[0]


def test_a_destination_outside_the_package_is_not_resolved(tmp_path: Path) -> None:
    """Scope control: the comparison covers this package's directory only.

    Recorded as a test rather than a comment because widening the resolver to
    every force-include destination would make it silently answer for files
    the parity comparison never looks at.
    """
    gate = _load_gate()
    root = _stage(tmp_path, with_source=True)
    (root / "pyproject.toml").write_text(
        PYPROJECT.read_text().replace(
            '"omnibase_core/data/gitignore-baseline.yaml"',
            '"somewhere_else/data/gitignore-baseline.yaml"',
        )
    )
    errors: list[str] = []

    assert gate._force_included_files(root, "omnibase_core", errors) == {}  # type: ignore[attr-defined]
    assert errors == []
