# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""release.yml resolves release evidence from where evidence lives after S6.

Why this file exists
--------------------
S6 of the change-control retirement (omnibase_infra#4792) deleted this
repository's autobind caller, so nothing writes a citation line into a release
PR's body any more. ``release.yml`` still read its evidence from that body, so
v0.38.70 (release PR #4845) was tagged and then stopped before the PyPI publish:
every release after S6 would have needed a hand-written companion plus a hand
edit of a merged PR body.

The step now reads evidence from where it lives:

1. ``contracts/<ticket>.yaml`` at the release commit, with a non-empty
   ``dod_evidence`` list (the in-repo contract the S6 cut-over moved to).
2. A PASS change-control receipt on onex_change_control's default branch, bound to the
   release PR by number, ticket and head sha.

and refuses (exit 1, before the build) when neither exists. These tests run the
shipped step, lifted from the workflow between its delimiters, under bash with a
stub ``gh`` that serves canned API responses, inside a throwaway git repository
standing in for the release checkout.

Ticket: OMN-20959
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RELEASE = _REPO_ROOT / ".github" / "workflows" / "release.yml"
_CASCADE = _REPO_ROOT / ".github" / "workflows" / "dependency-cascade.yml"

_OPEN = "# >>> OMN-20959 release evidence >>>"
_CLOSE = "# <<< OMN-20959 release evidence <<<"

_REPOSITORY = "OmniNode-ai/omnibase_infra"
_OCC = "repos/OmniNode-ai/onex_change_control"
_TICKET = "OMN-18595"
_PR = 4845
_PR_HEAD = "8ab93772c7f6e70974af001cb08e8294c2d77d27"
_OCC_COMMIT = "f403414d4cffe628193807e090779ca48ee2f090"
_RECEIPT_DIR = f"drift/dod_receipts/{_TICKET}/dod-OmniNode-ai-omnibase_infra-pr-{_PR}"

_STUB_GH = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, sys
    args = sys.argv[1:]
    with open(os.environ["FAKE_GH_LOG"], "a", encoding="utf-8") as log:
        log.write(" ".join(args) + "\\n")
    if len(args) != 2 or args[0] != "api":
        sys.stderr.write("stub gh: unsupported call " + " ".join(args) + "\\n")
        sys.exit(2)
    routes = json.load(open(os.environ["FAKE_GH_ROUTES"], encoding="utf-8"))
    if args[1] not in routes:
        sys.stderr.write("gh: Not Found (HTTP 404)\\n")
        sys.exit(1)
    sys.stdout.write(json.dumps(routes[args[1]]))
    """
)


def _program() -> str:
    """Lift the delimited resolver out of the shipped workflow, dedented."""
    body = _RELEASE.read_text(encoding="utf-8")
    start = body.rindex("\n", 0, body.index(_OPEN)) + 1
    end = body.index(_CLOSE, start)
    return textwrap.dedent(body[start:end])


def _tool_path() -> str:
    """A PATH holding only the directories of the tools these tests run.

    Every subprocess here gets a full replacement environment built from this,
    never a copy of the ambient one, so no inherited GIT_DIR / GIT_WORK_TREE can
    point a git call at the invoking repository (OMN-18434).
    """
    dirs: list[str] = []
    for tool in ("git", "bash", "python3", "sed", "mktemp"):
        found = shutil.which(tool)
        assert found, f"{tool} is required to run the shipped resolver"
        parent = str(Path(found).parent)
        if parent not in dirs:
            dirs.append(parent)
    return os.pathsep.join(dirs)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": _tool_path(),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "HOME": str(cwd),
        },
    ).stdout.strip()


def _pulls(
    head_sha: str,
    *,
    merged: bool = True,
    title: str = f"chore: release omnibase_infra 0.38.70 ({_TICKET})",
    ref: str = "automation/release-0.38.70-omn-18595",
    body: str = "",
    number: int = _PR,
    pr_head: str = _PR_HEAD,
) -> list[dict[str, object]]:
    return [
        {
            "number": _PR + 1,
            "state": "open",
            "merged_at": None,
            "title": "chore: an open PR on the same commit (OMN-99999)",
            "head": {"sha": head_sha, "ref": "jonah/omn-99999-open"},
            "body": "",
        },
        {
            "number": number,
            "state": "closed",
            "merged_at": "2026-10-11T05:00:00Z" if merged else None,
            "title": title,
            "head": {"sha": pr_head, "ref": ref},
            "body": body,
        },
    ]


def _receipt(
    *,
    status: str = "PASS",
    commit_sha: str = _PR_HEAD,
    pr_number: int = _PR,
    ticket: str = _TICKET,
) -> dict[str, object]:
    text = textwrap.dedent(
        f"""\
        ---
        schema_version: 1.0.0
        ticket_id: {ticket}
        evidence_item_id: dod-OmniNode-ai-omnibase_infra-pr-{_PR}
        check_type: command
        status: {status}
        commit_sha: {commit_sha}
        pr_number: {pr_number}
        branch: automation/release-0.38.70-omn-18595
        exit_code: 0
        """
    )
    return {
        "type": "file",
        "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
    }


def _occ_routes(receipt: dict[str, object]) -> dict[str, object]:
    path = f"{_RECEIPT_DIR}/command.yaml"
    return {
        f"{_OCC}/contents/{_RECEIPT_DIR}": [
            {"type": "file", "name": "command.yaml", "path": path}
        ],
        f"{_OCC}/contents/{path}": receipt,
        f"{_OCC}/commits?path={_RECEIPT_DIR}&per_page=1": [{"sha": _OCC_COMMIT}],
    }


def _run(
    tmp_path: Path,
    *,
    contract: str | None = None,
    extra_routes: dict[str, object] | None = None,
    pulls_kwargs: dict[str, Any] | None = None,
) -> tuple[int, str, dict[str, str], str]:
    """Run the shipped resolver; return (exit, output, GITHUB_OUTPUT, gh log)."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "-q")
    (checkout / "CHANGELOG.md").write_text("## v0.38.70\n", encoding="utf-8")
    if contract is not None:
        (checkout / "contracts").mkdir()
        (checkout / "contracts" / f"{_TICKET}.yaml").write_text(
            contract, encoding="utf-8"
        )
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-q", "-m", "release")
    head = _git(checkout, "rev-parse", "HEAD")

    routes: dict[str, object] = {
        f"repos/{_REPOSITORY}/commits/{head}/pulls": _pulls(
            head, **(pulls_kwargs or {})
        )
    }
    routes.update(extra_routes or {})

    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "gh"
    stub.write_text(_STUB_GH, encoding="utf-8")
    stub.chmod(0o755)
    routes_file = tmp_path / "routes.json"
    routes_file.write_text(json.dumps(routes), encoding="utf-8")
    log = tmp_path / "gh.log"
    log.touch()
    out = tmp_path / "gh_output"
    out.touch()

    result = subprocess.run(
        ["bash", "-c", _program()],
        cwd=checkout,
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": f"{bindir}{os.pathsep}{_tool_path()}",
            "HOME": str(tmp_path),
            "GITHUB_OUTPUT": str(out),
            "REPOSITORY": _REPOSITORY,
            "FAKE_GH_ROUTES": str(routes_file),
            "FAKE_GH_LOG": str(log),
        },
    )
    parsed: dict[str, str] = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            parsed[key] = value
    parsed["__head__"] = head
    return (
        result.returncode,
        result.stdout + result.stderr,
        parsed,
        log.read_text(encoding="utf-8"),
    )


_CONTRACT_WITH_EVIDENCE = textwrap.dedent(
    """\
    schema_version: "1.0.0"
    ticket_id: OMN-18595
    dod_evidence:
      - id: dod-release-cut-changelog
        description: the released version has a changelog section
        source: manual
        checks:
          - check_type: command
            check_value: grep -c '^## v' CHANGELOG.md
    """
)


class TestReleaseEvidenceResolvesWithoutABodyCitation:
    def test_release_evidence_from_the_in_repo_contract(self, tmp_path: Path) -> None:
        """AC1: no body citation, an in-repo contract with dod_evidence -> resolves."""
        code, output, parsed, log = _run(tmp_path, contract=_CONTRACT_WITH_EVIDENCE)
        assert code == 0, output
        assert parsed["ticket"] == _TICKET
        assert parsed["evidence_source"] == parsed["__head__"], (
            "the in-repo contract's evidence is the release commit that carries it"
        )
        assert parsed["evidence_kind"] == f"repo-contract:contracts/{_TICKET}.yaml"
        assert "onex_change_control" not in log, (
            "a repository with in-repo evidence must not depend on change control"
        )

    def test_release_evidence_from_the_change_control_receipt(
        self, tmp_path: Path
    ) -> None:
        """The v0.38.70 shape: no in-repo contract, a merged receipt bound to the PR head."""
        code, output, parsed, _ = _run(tmp_path, extra_routes=_occ_routes(_receipt()))
        assert code == 0, output
        assert parsed["ticket"] == _TICKET
        assert parsed["evidence_source"] == _OCC_COMMIT
        assert parsed["evidence_kind"].startswith("change-control-receipt:")

    def test_release_evidence_ticket_falls_back_to_the_head_branch(
        self, tmp_path: Path
    ) -> None:
        code, output, parsed, _ = _run(
            tmp_path,
            contract=_CONTRACT_WITH_EVIDENCE,
            pulls_kwargs={"title": "chore: release omnibase_infra 0.38.70"},
        )
        assert code == 0, output
        assert parsed["ticket"] == _TICKET

    def test_release_evidence_ignores_a_citation_in_the_pr_body(
        self, tmp_path: Path
    ) -> None:
        """A body line is not evidence: with nothing else, the release still refuses."""
        stamp = "Evidence-" + "Source: OCC#13655"
        code, output, parsed, _ = _run(
            tmp_path, pulls_kwargs={"body": f"{stamp}\nEvidence-Ticket: {_TICKET}\n"}
        )
        assert code == 1, output
        assert "evidence_source" not in parsed

    def test_release_evidence_step_no_longer_reads_the_pr_body(self) -> None:
        program = _program()
        assert "Evidence-" + "Source:" not in program
        assert "--json body" not in program


@pytest.mark.live_contact("tests/ci/fixtures/release_evidence_v0_38_70_omn20959.json")
def test_release_evidence_resolves_the_recorded_v0_38_70_release(
    tmp_path: Path, recorded_response: dict[str, object]
) -> None:
    """The real v0.38.70 release: its PR record and its merged change-control receipt.

    The receipt bytes are the file onex_change_control's default branch carries
    for release PR #4845, and the PR record is the PR watcher's. The shipped
    resolver must bind them by number, ticket and head sha, and name the
    change-control commit that wrote the receipt as the evidence source.
    """
    recording = cast("dict[str, object]", recorded_response["response"])
    pr = cast("dict[str, object]", recording["release_pr"])
    receipt_path = cast("str", recording["receipt_path"])
    receipt_dir = receipt_path.rsplit("/", 1)[0]
    occ_commit = cast("str", recording["receipt_dir_last_commit"])
    receipt_bytes = cast("str", recording["receipt_yaml"]).encode("utf-8")
    routes: dict[str, object] = {
        f"{_OCC}/contents/{receipt_dir}": [
            {
                "type": "file",
                "name": receipt_path.rsplit("/", 1)[1],
                "path": receipt_path,
            }
        ],
        f"{_OCC}/contents/{receipt_path}": {
            "type": "file",
            "content": base64.b64encode(receipt_bytes).decode("ascii"),
        },
        f"{_OCC}/commits?path={receipt_dir}&per_page=1": [{"sha": occ_commit}],
    }
    code, output, parsed, _ = _run(
        tmp_path,
        extra_routes=routes,
        pulls_kwargs={
            "number": pr["number"],
            "pr_head": pr["head_sha"],
            "title": pr["title"],
            "ref": pr["head_ref"],
        },
    )
    assert code == 0, output
    assert parsed["ticket"] == "OMN-18595"
    assert parsed["evidence_source"] == occ_commit
    assert parsed["evidence_kind"] == f"change-control-receipt:{receipt_path}"


class TestReleaseEvidenceRefuses:
    """AC3: the OMN-16286 refusal still fires when neither form exists."""

    def test_release_evidence_refuses_with_neither_form(self, tmp_path: Path) -> None:
        code, output, parsed, _ = _run(tmp_path)
        assert code == 1, output
        assert "OMN-16286" in output and "has no evidence" in output
        assert "evidence_source" not in parsed

    def test_release_evidence_refuses_an_empty_contract(self, tmp_path: Path) -> None:
        empty = 'schema_version: "1.0.0"\nticket_id: OMN-18595\ndod_evidence: []\n'
        code, output, _, _ = _run(tmp_path, contract=empty)
        assert code == 1, output

    def test_release_evidence_refuses_a_contract_without_dod_evidence(
        self, tmp_path: Path
    ) -> None:
        bare = 'schema_version: "1.0.0"\nticket_id: OMN-18595\nrequirements: []\n'
        code, output, _, _ = _run(tmp_path, contract=bare)
        assert code == 1, output

    @pytest.mark.parametrize(
        "receipt",
        [
            pytest.param(_receipt(status="FAIL"), id="failing-receipt"),
            pytest.param(_receipt(commit_sha="0" * 40), id="other-head"),
            pytest.param(_receipt(pr_number=_PR + 7), id="other-pr"),
            pytest.param(_receipt(ticket="OMN-1"), id="other-ticket"),
        ],
    )
    def test_release_evidence_refuses_an_unbound_receipt(
        self, tmp_path: Path, receipt: dict[str, object]
    ) -> None:
        code, output, parsed, _ = _run(tmp_path, extra_routes=_occ_routes(receipt))
        assert code == 1, output
        assert "evidence_source" not in parsed

    def test_release_evidence_refuses_without_a_merged_pr(self, tmp_path: Path) -> None:
        code, output, _, _ = _run(
            tmp_path,
            contract=_CONTRACT_WITH_EVIDENCE,
            pulls_kwargs={"merged": False},
        )
        assert code == 1, output
        assert "MERGED PR" in output

    def test_release_evidence_refuses_without_a_ticket(self, tmp_path: Path) -> None:
        code, output, _, _ = _run(
            tmp_path,
            contract=_CONTRACT_WITH_EVIDENCE,
            pulls_kwargs={"title": "chore: release", "ref": "automation/release"},
        )
        assert code == 1, output
        assert "names no OMN ticket" in output


class TestDependencyCascadeEvidence:
    """AC2 (first half): the cascade accepts both shapes the release now emits."""

    @staticmethod
    def _cascade_vars(evidence_source: str, tmp_path: Path) -> int:
        body = _CASCADE.read_text(encoding="utf-8")
        start = body.rindex("\n", 0, body.index("# >>> OMN-18596 bump branch >>>")) + 1
        end = body.index("# <<< OMN-18596 bump branch <<<", start)
        out = tmp_path / "gh_output"
        out.touch()
        return subprocess.run(
            ["bash", "-c", textwrap.dedent(body[start:end])],
            capture_output=True,
            text=True,
            check=False,
            env={
                "PATH": _tool_path(),
                "GITHUB_OUTPUT": str(out),
                "PACKAGE": "omnibase_infra",
                "VERSION": "v0.38.70",
                "TICKET": _TICKET,
                "EVIDENCE_SOURCE": evidence_source,
            },
        ).returncode

    @pytest.mark.parametrize(
        "evidence_source",
        [
            pytest.param("1" * 40, id="release-commit"),
            pytest.param(_OCC_COMMIT, id="change-control-commit"),
        ],
    )
    def test_dependency_cascade_evidence_accepts_the_release_shapes(
        self, tmp_path: Path, evidence_source: str
    ) -> None:
        assert self._cascade_vars(evidence_source, tmp_path) == 0
