# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replays for the Contract Compliance evidence guards (OMN-20135).

Two guards, two real incidents, every byte captured rather than retyped.

1. ``scripts/ci/run_contract_compliance_with_evidence.py``. On 2026-09-30 the
   Contract Compliance Check on omnibase_infra#4325 (job 109722692581) resolved
   ticket OMN-17427, looked for its contract in the change-control checkout
   pinned at 7352cb2d (2026-07-12), found none, printed that the PR was not
   blocked and passed with nothing executed. The contract existed; it was only
   newer than the pin. The artifact is that job's log, fetched from the jobs
   API. The real guard, handed the contracts directory the log says the job
   read (no ``OMN-17427.yaml`` in it), must refuse. The discriminator hands the
   same guard the contract as the PR's evidence commit carries it (the
   ``contracts/OMN-17427.yaml`` blob at change-control 67e79ddf, the merge of
   the companion omnibase_infra#4325 cites) and it must go on to run the checks.

2. ``scripts/ci/resolve_contract_compliance_evidence.py``. On 2026-07-26 an
   change-control hygiene sweep closed companion PRs whose product PRs had
   already merged (OMN-15214). omniintelligence#817 merged citing OCC#5012,
   closed without merging, and nothing in a Contract Compliance Check that read
   contracts from a fixed pin could see that. The artifacts are both PRs as the
   REST API serves them. The real resolver, reading that product PR's body and
   that companion's state, must refuse. The discriminator is omnibase_infra#4325
   and its merged companion OCC#11857, which must resolve to the merge commit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _ROOT / "tests/fixtures/omn20135"
_WRAPPER = _ROOT / "scripts/ci/run_contract_compliance_with_evidence.py"
_RESOLVER = _ROOT / "scripts/ci/resolve_contract_compliance_evidence.py"

_JOB_LOG = _FIXTURES / "contract-compliance-job-109722692581.log.captured"
_CONTRACT = _FIXTURES / "OMN-17427.yaml.captured"
_PR_817 = _FIXTURES / "omniintelligence-pulls-817.json.captured"
_OCC_5012 = _FIXTURES / "onex_change_control-pulls-5012.json.captured"
_PR_4325 = _FIXTURES / "omnibase_infra-pulls-4325.json.captured"
_OCC_11857 = _FIXTURES / "onex_change_control-pulls-11857.json.captured"

_SHA256 = {
    _JOB_LOG: "43959a0b14035aedddeb3d04ea31b5031cbbb50c8f438865a18cef918a9181c5",
    _CONTRACT: "75b0982619602d34c5bf156fb6748ff32a9c084ed5afea216af78f34f99e2f77",
    _PR_817: "9d61f37e6e448e6b5fec172565500d2ab6317a3f640875081939f3832f5a27b2",
    _OCC_5012: "5fc6d1f1e2fbc56221ed9668c61b8bf6323d511878d8e0a66a10ea912a27ac4a",
    _PR_4325: "5b67705870d59f637f3d98ded401fb8f3aa88b52904cf1523f62fadc4e1d6022",
    _OCC_11857: "e7b2270033d6dbe75fe9be132ca943eea243ba227d50aa85cce75d034b6b7e40",
}

_OCC_REPO = "OmniNode-ai/onex_change_control"


@pytest.mark.parametrize("fixture", sorted(_SHA256, key=str))
def test_the_fixtures_are_the_captured_bytes(fixture: Path) -> None:
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == _SHA256[fixture]


def _tool_path(bindir: Path) -> str:
    tool_dirs = sorted(
        {
            str(Path(found).parent)
            for found in (shutil.which("bash"), shutil.which("env"))
            if found is not None
        }
    )
    return os.pathsep.join([str(bindir), *tool_dirs, "/usr/bin", "/bin"])


# --- 1. the wrapper, on the job that passed a PR whose contract it never read --


def _what_the_job_read() -> tuple[str, str]:
    """The ticket and the contract path from the captured log, and its verdict."""
    log = _JOB_LOG.read_text(encoding="utf-8", errors="replace")
    ticket = re.search(r"\[INFO\] Ticket: (OMN-\d+), PR: #4325", log)
    missing = re.search(r"\[WARN\] No contract at (\S+)\. .*PR not blocked\.", log)
    assert ticket is not None and missing is not None, "not the incident log"
    return ticket.group(1), missing.group(1)


def _checker(tmp_path: Path, ticket: str) -> Path:
    """A checker whose ticket resolver answers what the real run resolved."""
    checker = tmp_path / "checker"
    module = checker / "src/onex_change_control/scripts"
    module.mkdir(parents=True)
    for package in (
        checker / "src/onex_change_control/__init__.py",
        module / "__init__.py",
    ):
        package.write_text("", encoding="utf-8")
    (module / "contract_compliance_check.py").write_text(
        f"def _extract_ticket_id(pr_number, repo):\n    return {ticket!r}\n",
        encoding="utf-8",
    )
    return checker


def _run_wrapper(
    tmp_path: Path, evidence: Path, ticket: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    checker = _checker(tmp_path, ticket)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "uv.log"
    uv = bindir / "uv"
    uv.write_text(
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > {log}\n", encoding="utf-8"
    )
    uv.chmod(0o755)
    result = subprocess.run(
        [
            sys.executable,
            str(_WRAPPER),
            "--pr",
            "4325",
            "--repo",
            "OmniNode-ai/omnibase_infra",
            "--checker-dir",
            str(checker),
            "--evidence-contracts-dir",
            str(evidence),
            "--workspace",
            str(tmp_path),
            "--legacy-allowlist",
            str(checker / "allowlist.txt"),
            "--deferred-record",
            str(tmp_path / "record.json"),
        ],
        env={"PATH": _tool_path(bindir)},
        capture_output=True,
        text=True,
        check=False,
    )
    return result, log


def test_the_real_guard_refuses_the_pr_the_pinned_checkout_passed(
    tmp_path: Path,
) -> None:
    ticket, missing = _what_the_job_read()
    assert ticket == "OMN-17427"
    assert missing.endswith(f"/onex_change_control/contracts/{ticket}.yaml")
    # The directory the job read, as the log reports it: no contract for the
    # ticket. The guard must refuse where the pinned runner said "not blocked".
    pinned_contracts = tmp_path / "onex_change_control/contracts"
    pinned_contracts.mkdir(parents=True)
    result, uv_log = _run_wrapper(tmp_path, pinned_contracts, ticket)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "lacks the resolved ticket contract" in result.stderr
    assert f"{ticket}.yaml" in result.stderr
    assert not uv_log.exists(), "no check may run once the contract is missing"


def test_the_same_guard_runs_the_contract_the_evidence_commit_carries(
    tmp_path: Path,
) -> None:
    ticket, _ = _what_the_job_read()
    evidence = tmp_path / "onex_change_control_evidence/contracts"
    evidence.mkdir(parents=True)
    shutil.copyfile(_CONTRACT, evidence / f"{ticket}.yaml")
    result, uv_log = _run_wrapper(tmp_path, evidence, ticket)
    assert result.returncode == 0, result.stdout + result.stderr
    invocation = uv_log.read_text(encoding="utf-8")
    assert f"--contracts-dir {evidence}" in invocation
    assert "defer_test_passes_driver.py" in invocation


# --- 2. the resolver, on a PR whose companion was closed without merging ------


def _pr_view(rest: dict[str, object]) -> dict[str, object]:
    """``gh pr view --json state,headRefOid,mergeCommit,body`` from a REST PR."""
    merged = bool(rest.get("merged"))
    state = "MERGED" if merged else ("OPEN" if rest["state"] == "open" else "CLOSED")
    head = rest["head"]
    assert isinstance(head, dict)
    return {
        "state": state,
        "headRefOid": head["sha"],
        "mergeCommit": {"oid": rest["merge_commit_sha"]} if merged else None,
        "body": rest["body"],
    }


def _resolve(
    tmp_path: Path, *, repo: str, product: Path, companion: Path
) -> tuple[subprocess.CompletedProcess[str], str]:
    product_rest = json.loads(product.read_text(encoding="utf-8"))
    companion_rest = json.loads(companion.read_text(encoding="utf-8"))
    product_view = json.dumps({"body": product_rest["body"]})
    companion_view = json.dumps(
        {
            k: v
            for k, v in _pr_view(companion_rest).items()
            if k in {"state", "headRefOid", "mergeCommit"}
        }
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "product.json").write_text(product_view, encoding="utf-8")
    (bindir / "companion.json").write_text(companion_view, encoding="utf-8")
    gh = bindir / "gh"
    gh.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'case "$*" in\n'
        f'  *"pr view {product_rest["number"]} --repo {repo} --json body"*)'
        f" cat {bindir / 'product.json'} ;;\n"
        f'  *"pr view {companion_rest["number"]} --repo {_OCC_REPO}"*)'
        f" cat {bindir / 'companion.json'} ;;\n"
        '  *) echo "unexpected gh $*" >&2; exit 8 ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    output = tmp_path / "github-output"
    result = subprocess.run(
        [
            sys.executable,
            str(_RESOLVER),
            "--repo",
            repo,
            "--event-name",
            "pull_request",
            "--commit-sha",
            str(product_rest["merge_commit_sha"]),
            "--pr-number",
            str(product_rest["number"]),
            "--github-output",
            str(output),
        ],
        env={"PATH": _tool_path(bindir)},
        capture_output=True,
        text=True,
        check=False,
    )
    written = output.read_text(encoding="utf-8") if output.exists() else ""
    return result, written


def test_the_real_resolver_refuses_the_closed_companion_a_merged_pr_cited(
    tmp_path: Path,
) -> None:
    product = json.loads(_PR_817.read_text(encoding="utf-8"))
    companion = json.loads(_OCC_5012.read_text(encoding="utf-8"))
    assert product["merged"] is True, "the product PR merged on this evidence"
    assert re.search(r"^Evidence-Source: OCC#5012$", product["body"], re.MULTILINE)
    assert companion["state"] == "closed" and companion["merged"] is False
    result, written = _resolve(
        tmp_path,
        repo="OmniNode-ai/omniintelligence",
        product=_PR_817,
        companion=_OCC_5012,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "OCC PR #5012 is 'CLOSED'" in result.stderr
    assert written == ""


def test_the_same_resolver_resolves_a_merged_companion_to_its_merge_commit(
    tmp_path: Path,
) -> None:
    companion = json.loads(_OCC_11857.read_text(encoding="utf-8"))
    assert companion["merged"] is True
    result, written = _resolve(
        tmp_path,
        repo="OmniNode-ai/omnibase_infra",
        product=_PR_4325,
        companion=_OCC_11857,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert written == f"pr_number=4325\nocc_sha={companion['merge_commit_sha']}\n"
