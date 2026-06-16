# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Enforcement ratchet for OMN-12965 — runtime image identity labels.

The deep dive found the main runtime image stamped
``org.opencontainers.image.version=0.1.0`` with a blank
``org.opencontainers.image.revision`` after workspace rebuilds. A blank identity
degrades EVERY proof packet, because the runtime SHA + image digest are required
citations in accepted evidence.

Root cause: the operator rebuild path ``onex up --build`` (``cmd_up`` in
``omnibase_infra.docker.catalog.cli``) passed only ``GIT_SHA`` as a build arg.
The runtime-stage OCI labels read ``VCS_REF`` (-> blank revision) and
``RUNTIME_VERSION`` (-> placeholder ``0.1.0``), neither of which were passed.

These tests pin the fix so it cannot regress:

1. ``_image_identity_build_args`` stamps the full identity quad
   (GIT_SHA / VCS_REF / RUNTIME_VERSION / BUILD_DATE).
2. ``_image_identity_build_args`` fails fast when the git revision cannot be
   resolved, rather than stamping a blank/``unknown`` revision.
3. ``_runtime_version`` resolves the real package version (never the Dockerfile
   placeholder).
4. ``Dockerfile.runtime`` carries a workspace-mode guard that fails the build
   when VCS_REF or RUNTIME_VERSION would be blank/placeholder.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from omnibase_infra.docker.catalog import cli

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.runtime"

_IDENTITY_ARGS = ("GIT_SHA", "VCS_REF", "RUNTIME_VERSION", "BUILD_DATE")


def _parse_build_args(args: list[str]) -> dict[str, str]:
    """Collapse a ``--build-arg KEY=VALUE`` list into a dict."""
    parsed: dict[str, str] = {}
    it = iter(args)
    for token in it:
        if token == "--build-arg":
            key, _, value = next(it).partition("=")
            parsed[key] = value
    return parsed


def test_runtime_version_matches_pyproject() -> None:
    """_runtime_version must return the real package version, not the placeholder."""
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = str(data["project"]["version"]).strip()
    assert cli._runtime_version() == expected
    assert cli._runtime_version() != "0.1.0", (
        "runtime version must not be the Dockerfile placeholder 0.1.0 (OMN-12965)"
    )


def test_runtime_image_bakes_headless_codex_cli() -> None:
    """Runtime image must carry codex for cli://codex delegation."""
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG CODEX_CLI_VERSION=0.140.0" in dockerfile
    assert "FROM node:22-bookworm-slim AS node-bin" in dockerfile
    assert 'npm install -g "@openai/codex@${CODEX_CLI_VERSION}"' in dockerfile
    assert "COPY --from=node-bin /usr/local/bin/node /usr/local/bin/node" in dockerfile
    assert (
        "COPY --from=node-bin /usr/local/lib/node_modules /usr/local/lib/node_modules"
    ) in dockerfile
    assert (
        "ln -sf ../lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex"
    ) in dockerfile
    assert "codex --version" in dockerfile
    assert "CODEX_HOME=/home/omniinfra/.codex" in dockerfile
    assert "install -d -o omniinfra -g omniinfra" in dockerfile
    assert "/home/omniinfra/.codex" in dockerfile


def test_identity_build_args_stamp_full_quad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The identity quad must be present so the runtime-stage labels populate."""
    monkeypatch.setattr(cli, "_current_git_sha", lambda: "deadbeefcafef00d")
    parsed = _parse_build_args(cli._image_identity_build_args())

    for key in _IDENTITY_ARGS:
        assert key in parsed, f"missing --build-arg {key} (OMN-12965)"

    # VCS_REF (-> org.opencontainers.image.revision) must carry the real SHA.
    assert parsed["VCS_REF"] == "deadbeefcafef00d"
    assert parsed["GIT_SHA"] == "deadbeefcafef00d"
    # RUNTIME_VERSION (-> org.opencontainers.image.version) must be the real version.
    assert parsed["RUNTIME_VERSION"] == cli._runtime_version()
    assert parsed["RUNTIME_VERSION"] != "0.1.0"
    # BUILD_DATE (-> org.opencontainers.image.created) must be an RFC3339 stamp.
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", parsed["BUILD_DATE"]
    ), parsed["BUILD_DATE"]


@pytest.mark.parametrize("bad_sha", ["", "unknown"])
def test_identity_build_args_fail_fast_on_unresolved_revision(
    monkeypatch: pytest.MonkeyPatch, bad_sha: str
) -> None:
    """A blank/unknown revision must abort the build, not ship blank identity."""
    monkeypatch.setattr(cli, "_current_git_sha", lambda: bad_sha)
    with pytest.raises(RuntimeError, match="git revision"):
        cli._image_identity_build_args()


def test_cmd_up_build_path_uses_identity_args() -> None:
    """cmd_up's --build branch must call the identity-arg helper, not bare GIT_SHA.

    Static guard scoped to the ``cmd_up`` function body: the build branch must
    spread ``_image_identity_build_args()`` and must NOT hand-roll a lone
    ``GIT_SHA`` build-arg (the pre-fix regression that left VCS_REF +
    RUNTIME_VERSION unstamped → blank-identity image, OMN-12965).
    """
    source = Path(cli.__file__).read_text(encoding="utf-8")
    body = _function_body(source, "cmd_up")
    assert "_image_identity_build_args()" in body, (
        "cmd_up --build must stamp the identity quad via _image_identity_build_args()"
    )
    assert "GIT_SHA=" not in body, (
        "cmd_up must not hand-roll a GIT_SHA build-arg; delegate to "
        "_image_identity_build_args() so VCS_REF + RUNTIME_VERSION are stamped "
        "(OMN-12965)."
    )


def _function_body(source: str, name: str) -> str:
    """Return the source text of a top-level ``def name(...)`` block."""
    lines = source.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(f"def {name}(")),
        None,
    )
    assert start is not None, f"function {name} not found in cli.py"
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i] and not lines[i][0].isspace()
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_dockerfile_workspace_identity_guard_present() -> None:
    """Dockerfile.runtime must fail workspace builds with blank/placeholder identity."""
    text = _DOCKERFILE.read_text(encoding="utf-8")
    collapsed = re.sub(r"\\\n\s*", " ", text)

    # The runtime stage must label revision from VCS_REF and version from RUNTIME_VERSION.
    assert 'org.opencontainers.image.revision="${VCS_REF}"' in collapsed
    assert 'org.opencontainers.image.version="${RUNTIME_VERSION}"' in collapsed

    # Workspace-mode guard: reject blank/unknown VCS_REF.
    guard = re.search(
        r'if \[ "\$\{BUILD_SOURCE\}" = "workspace" \];.*?VCS_REF.*?'
        r"exit 64.*?RUNTIME_VERSION.*?0\.1\.0.*?exit 64",
        collapsed,
        re.DOTALL,
    )
    assert guard, (
        "Dockerfile.runtime must guard workspace builds against blank VCS_REF and "
        "placeholder RUNTIME_VERSION=0.1.0 (OMN-12965)."
    )
