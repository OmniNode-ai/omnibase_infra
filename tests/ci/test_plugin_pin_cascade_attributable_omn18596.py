# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PR-time gate: the plugin pin cascade opens an attributable PR on the right base.

``plugin-pin-cascade.yml`` is the sibling of ``dependency-cascade.yml`` that the
OMN-18596 convergence did not reach. It opened its PR with no ticket in the
title or head ref, so occ-autobind had nothing to bind a change-control
companion to, and with a literal ``--base main`` while the checkout cut the
branch from the default branch ``dev``. omnibase_infra#4168 is the specimen: a
three-line ``docker/Dockerfile.runtime`` change rendered as a 100-file
``dev..main`` diff, with no companion and no way to get one.

Pinned by EXECUTION where it matters: the delimited input-validation and
base-resolver programs are lifted out of the shipped YAML and run, the same way
``test_dependency_cascade_base_branch_omn18596.py`` pins the sibling.

Ticket: OMN-18596
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "plugin-pin-cascade.yml"

_INPUTS_OPEN = "# >>> OMN-18596 plugin pin inputs >>>"
_INPUTS_CLOSE = "# <<< OMN-18596 plugin pin inputs <<<"
_BASE_OPEN = "# >>> OMN-18596 plugin pin base resolver >>>"
_BASE_CLOSE = "# <<< OMN-18596 plugin pin base resolver <<<"


def _text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def _program(open_marker: str, close_marker: str) -> str:
    body = _text()
    start = body.index(open_marker)
    end = body.index(close_marker, start)
    lines = [
        line[10:] if line.startswith(" " * 10) else line
        for line in body[start:end].splitlines()
    ]
    return re.sub(r"\$\{\{[^}]*\}\}", "templated", "\n".join(lines))


def _run(
    program: str, cwd: Path, env: dict[str, str], out: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", program],
        cwd=cwd,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "GITHUB_OUTPUT": str(out),
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def _outputs(out: Path) -> dict[str, str]:
    if not out.exists():
        return {}
    return dict(
        line.split("=", 1) for line in out.read_text().splitlines() if "=" in line
    )


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )


def _standing_ticket() -> str:
    data = yaml.safe_load(_text())
    return str(data["env"]["PLUGIN_PIN_CASCADE_STANDING_TICKET"])


class TestAttributable:
    def test_a_standing_ticket_is_declared_as_workflow_config(self) -> None:
        assert re.fullmatch(r"OMN-\d+", _standing_ticket())

    def test_dispatch_without_a_ticket_resolves_the_standing_ticket(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        result = _run(
            _program(_INPUTS_OPEN, _INPUTS_CLOSE),
            tmp_path,
            {
                "EVENT_NAME": "repository_dispatch",
                "DISPATCH_PACKAGE": "omninode-memory",
                "DISPATCH_VERSION": "v0.18.3",
                "DISPATCH_SOURCE_REPO": "OmniNode-ai/omnimemory",
                "DISPATCH_TICKET": "",
                "PLUGIN_PIN_CASCADE_STANDING_TICKET": _standing_ticket(),
            },
            out,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert _outputs(out)["ticket"] == _standing_ticket()

    def test_a_dispatched_ticket_overrides_the_standing_one(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        result = _run(
            _program(_INPUTS_OPEN, _INPUTS_CLOSE),
            tmp_path,
            {
                "EVENT_NAME": "repository_dispatch",
                "DISPATCH_PACKAGE": "omninode-memory",
                "DISPATCH_VERSION": "0.18.3",
                "DISPATCH_TICKET": "OMN-12345",
                "PLUGIN_PIN_CASCADE_STANDING_TICKET": _standing_ticket(),
            },
            out,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert _outputs(out)["ticket"] == "OMN-12345"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("DISPATCH_TICKET", "OMN-1; rm -rf /"),
            ("DISPATCH_PACKAGE", "omninode-memory$(id)"),
            ("DISPATCH_VERSION", "latest"),
        ],
    )
    def test_malformed_input_refuses(
        self, tmp_path: Path, field: str, value: str
    ) -> None:
        env = {
            "EVENT_NAME": "repository_dispatch",
            "DISPATCH_PACKAGE": "omninode-memory",
            "DISPATCH_VERSION": "0.18.3",
            "DISPATCH_TICKET": "",
            "PLUGIN_PIN_CASCADE_STANDING_TICKET": _standing_ticket(),
            field: value,
        }
        result = _run(
            _program(_INPUTS_OPEN, _INPUTS_CLOSE), tmp_path, env, tmp_path / "out"
        )
        assert result.returncode != 0

    def test_title_head_ref_and_body_carry_the_ticket(self) -> None:
        text = _text()
        assert '${{ steps.inputs.outputs.ticket }})"' in text, "title cites the ticket"
        assert "-${TICKET_SLUG}" in text, "head ref carries the ticket"
        assert "Evidence-Ticket: ${{ steps.inputs.outputs.ticket }}" in text

    def test_no_dispatch_value_is_interpolated_into_a_script(self) -> None:
        for step_script in re.findall(r"run: \|\n((?:\s{10}.*\n)+)", _text()):
            assert "client_payload" not in step_script


class TestBaseIsResolvedNotWritten:
    def test_no_literal_base_branch(self) -> None:
        text = _text()
        assert not re.search(r"--base\s+(main|dev)\b", text)
        assert '--base "${{ steps.branch.outputs.base }}"' in text

    def test_resolver_reads_the_checkout_branch(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        root.mkdir()
        _git(root, "init", "--initial-branch", "trunk")
        _git(root, "config", "user.email", "t@example.invalid")
        _git(root, "config", "user.name", "t")
        (root / "f").write_text("x\n")
        _git(root, "add", "f")
        _git(root, "commit", "-m", "init")
        out = tmp_path / "out"
        result = _run(_program(_BASE_OPEN, _BASE_CLOSE), root, {}, out)
        assert result.returncode == 0, result.stderr
        assert _outputs(out)["base"] == "trunk"

    def test_a_detached_checkout_refuses(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        root.mkdir()
        _git(root, "init", "--initial-branch", "trunk")
        _git(root, "config", "user.email", "t@example.invalid")
        _git(root, "config", "user.name", "t")
        (root / "f").write_text("x\n")
        _git(root, "add", "f")
        _git(root, "commit", "-m", "init")
        _git(root, "checkout", "--detach")
        out = tmp_path / "out"
        result = _run(_program(_BASE_OPEN, _BASE_CLOSE), root, {}, out)
        assert result.returncode != 0
        assert "base" not in _outputs(out)
