# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lab pool leaves a pr-head receipt behind every proof (OMN-19566, T2 slice 2).

Every run of ``prepr_runtime_pool.py run`` now mints one ``pr-head`` receipt per
PR it proved, PASS, FAIL or INCONCLUSIVE, keyed by the exact head the clone
fetched, judged against the PR's registry profile, with the lane that ran it
distinct from the judge that minted it. These tests drive the driver against the
same fake hosts as its OMN-18893 tests and read the receipts back.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_PY = REPO_ROOT / "scripts" / "runtime_build" / "prepr_runtime_pool.py"
PROVE_SH = REPO_ROOT / "scripts" / "runtime_build" / "prepr_pool_prove.sh"
POOL_TESTS = Path(__file__).with_name("test_prepr_runtime_pool_omn18893.py")

pytestmark = pytest.mark.unit


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pool = _load("prepr_runtime_pool", POOL_PY)
fakes = _load("prepr_runtime_pool_fakes_omn19566", POOL_TESTS)
receipts = pool._receipts()
CFG = pool.load_pool_config()
NOW = dt.datetime(2026, 9, 28, 10, 0, tzinfo=dt.UTC)

HEAD = "1" * 40
DEV = "2" * 40
MERGE_BASE = "3" * 40
DIGEST = "sha256:" + "4" * 64
MOVED = "5" * 40

CLONE = (
    f"omnimarket#3042 fetched head {HEAD} expected {HEAD} match=yes\n"
    f"pr-subject omnimarket#3042 head={HEAD} base={DEV} merge-base={MERGE_BASE} "
    f"diff-digest={DIGEST}\n"
    f"omnimarket test-merge clean {'6' * 40} (dev {DEV})\n"
    "clone done"
)
LIVE = (
    "  omnibase-infra-local-omninode-runtime mentions node_lab_proof_receipt: 3\n"
    "  omnibase-infra-local-runtime-effects mentions node_lab_proof_receipt: 0\n"
)


def _params(tmp_path: Path) -> Path:
    p = tmp_path / "p.env"
    p.write_text(
        f"MARKET_PR=3042\nMARKET_HEAD={HEAD}\nTESTS=omnimarket:tests/test_x.py\n",
        encoding="utf-8",
    )
    return p


def _run(
    tmp_path: Path,
    *,
    clone: str = CLONE,
    live: str = LIVE,
    probe_extra: str = "",
    build: str | None = None,
    verifier: str | None = "prepr_runtime_pool.judge@test-mac",
) -> tuple[int, str, list[dict[str, Any]], list[dict[str, Any]]]:
    host = fakes.FakeHost()
    host.phases["clone"] = clone
    host.phases["probe"] = fakes.GOOD["probe"] + live + probe_extra
    if build is not None:
        host.phases["build"] = build
    out = tmp_path / "receipts"
    code, text = pool.run_proof(
        CFG,
        fakes.FakeTransport({"lab-101": host}),
        _params(tmp_path),
        "plan-item-I7",
        60,
        "lab-101",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
        receipt_dir=out,
        verifier_identity=verifier,
    )
    found = sorted(out.glob("*.receipt.json")) if out.exists() else []
    events = sorted(out.glob("*.event.json")) if out.exists() else []
    return (
        code,
        text,
        [json.loads(p.read_text()) for p in found],
        [json.loads(p.read_text()) for p in events],
    )


def test_a_pass_mints_an_accepted_receipt_keyed_by_the_exact_head(
    tmp_path: Path,
) -> None:
    code, text, found, events = _run(
        tmp_path,
        live="  omnibase-infra-local-omninode-runtime mentions node_x: 2\n",
    )
    assert code == pool.EXIT_PASS, text
    assert len(found) == 1 and len(events) == 1
    receipt = receipts.ModelLabPassReceipt.from_json(json.dumps(found[0]))
    assert receipt.sha == HEAD
    assert receipt.lane is receipts.EnumLabLane.PR_HEAD
    assert receipt.result is receipts.EnumLabPassResult.PASS
    subject = receipt.subject
    assert subject is not None
    assert subject.repo == "OmniNode-ai/omnimarket"
    assert subject.pr_number == 3042
    assert (subject.base_sha, subject.merge_base_sha) == (DEV, MERGE_BASE)
    assert subject.pr_diff_digest == DIGEST
    assert subject.profile_id == "omnimarket.runtime_image"
    assert subject.profile_version == "1"
    assert subject.handler_kind is receipts.EnumLabProofHandlerKind.RUNTIME_IMAGE
    assert subject.runner_identity == "plan-item-I7@lab-101"
    assert subject.verifier_identity == "prepr_runtime_pool.judge@test-mac"
    # the receipt speaks the profile vocabulary, so the profile can judge it
    names = {c.name for c in receipt.checks}
    assert {
        "subject_head_identity",
        "runtime_image_identity",
        "migration_gate_healthy",
        "runtime_main_healthy",
        "runtime_effects_healthy",
        "no_wiring_failures",
        "focused_tests",
        "changed_path_live",
    } <= names
    event = events[0]
    assert event["topic"] == "onex.evt.omnibase-infra.lab-proof-receipt.v1"
    assert event["verifier_token"] == "ACCEPTED"
    assert (
        f"receipt OmniNode-ai/omnimarket#3042@{HEAD}:omnimarket.runtime_image@1 "
        "result=PASS verifier=ACCEPTED"
    ) in text


def test_a_pass_with_no_changed_path_readback_is_refused_as_missing_a_check(
    tmp_path: Path,
) -> None:
    code, text, found, events = _run(tmp_path, live="")
    # the readback is the judge's and stays PASS; the receipt says what it lacks
    assert code == pool.EXIT_PASS, text
    assert found[0]["result"] == "PASS"
    assert events[0]["verifier_token"] == "MISSING_MANDATORY_CHECK"
    assert events[0]["missing_mandatory_checks"] == ["changed_path_live"]
    assert "verifier=MISSING_MANDATORY_CHECK" in text


def test_a_changed_path_no_runtime_mentions_fails_the_proof(tmp_path: Path) -> None:
    code, text, found, events = _run(
        tmp_path,
        live=(
            "  omnibase-infra-local-omninode-runtime mentions node_dead: 0\n"
            "  omnibase-infra-local-runtime-effects mentions node_dead: 0\n"
        ),
    )
    assert code == pool.EXIT_FAIL, text
    assert "changed path not live, no runtime log mentions: node_dead" in text
    assert found[0]["result"] == "FAIL"
    assert events[0]["failing_checks"] == ["changed_path_live"]
    assert events[0]["verifier_token"] == "RESULT_NOT_PASS"


def test_a_runner_that_is_its_own_verifier_is_refused(tmp_path: Path) -> None:
    _, text, _, events = _run(tmp_path, verifier="PLAN-ITEM-I7@lab-101")
    assert events[0]["verifier_token"] == "RUNNER_IS_VERIFIER"
    assert "verifier=RUNNER_IS_VERIFIER" in text


def test_the_default_verifier_is_never_the_runner(tmp_path: Path) -> None:
    _, _, found, _ = _run(tmp_path, verifier=None)
    subject = found[0]["subject"]
    assert re.fullmatch(
        r"prepr_runtime_pool\.judge@driver-[0-9a-f]{10}", subject["verifier_identity"]
    )
    assert subject["verifier_identity"] != subject["runner_identity"]
    # published on public repositories: no machine name, no LAN address
    assert socket.gethostname() not in json.dumps(found[0])
    assert subject["host"] == "lab-101"
    assert "192.168." not in json.dumps(found[0])


def test_a_failing_proof_still_leaves_a_fail_receipt(tmp_path: Path) -> None:
    # the judge reads the first 8085 line, so make the failing one the only one
    host_probe = fakes.GOOD["probe"].replace(
        "port 8085 HTTP 200 status healthy healthy True failed_handlers 0",
        "port 8085 HTTP 503 status unhealthy healthy False failed_handlers 2",
    )
    host = fakes.FakeHost()
    host.phases["clone"] = CLONE
    host.phases["probe"] = host_probe + LIVE
    out = tmp_path / "fail"
    code, _text = pool.run_proof(
        CFG,
        fakes.FakeTransport({"lab-101": host}),
        _params(tmp_path),
        "plan-item-I7",
        60,
        "lab-101",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
        receipt_dir=out,
        verifier_identity="judge@mac",
    )
    assert code == pool.EXIT_FAIL
    (event_path,) = out.glob("*.event.json")
    event = json.loads(event_path.read_text())
    assert event["result"] == "FAIL"
    assert "runtime_main_healthy" in event["failing_checks"]
    assert event["verifier_token"] == "RESULT_NOT_PASS"


def test_an_inconclusive_run_leaves_a_fail_receipt_of_indeterminate_checks(
    tmp_path: Path,
) -> None:
    code, _, found, _ = _run(tmp_path, build="build rc=1\nup rc=1")
    assert code == pool.EXIT_INCONCLUSIVE
    receipt = receipts.ModelLabPassReceipt.from_json(json.dumps(found[0]))
    assert receipt.result is receipts.EnumLabPassResult.FAIL
    outcome = {c.name: c.outcome.value for c in receipt.checks}
    assert outcome["subject_head_identity"] == "pass"
    assert outcome["stack_built"] == "indeterminate" or outcome["stack_built"] == "fail"
    assert outcome["runtime_main_healthy"] == "indeterminate"
    assert outcome["focused_tests"] == "indeterminate"


def test_a_head_that_moved_before_the_fetch_is_recorded_against_the_fetched_head(
    tmp_path: Path,
) -> None:
    moved = CLONE.replace(
        f"fetched head {HEAD} expected {HEAD} match=yes",
        f"fetched head {MOVED} expected {HEAD} match=NO",
    ).replace(f"head={HEAD}", f"head={MOVED}")
    code, _, found, events = _run(tmp_path, clone=moved)
    assert code == pool.EXIT_FAIL
    assert found[0]["sha"] == MOVED
    assert events[0]["head_sha"] == MOVED
    assert "subject_head_identity" in events[0]["failing_checks"]
    # the key never names the head the params asked for
    assert HEAD not in events[0]["receipt_key"]


def test_an_older_clone_with_no_subject_line_mints_nothing_and_says_so(
    tmp_path: Path,
) -> None:
    older = "\n".join(line for line in CLONE.splitlines() if "pr-subject" not in line)
    code, text, found, _ = _run(tmp_path, clone=older)
    assert code == pool.EXIT_PASS
    assert found == []
    assert "receipt: no pr-subject line in the clone output" in text


def test_every_runtime_profile_the_receipt_names_exists_in_the_registry() -> None:
    for repo in ("omnimarket", "omnibase_infra"):
        profile = pool.runtime_profile(repo)
        assert profile is not None
        assert "changed_path_live" in profile.mandatory_checks
    assert pool.runtime_profile("omnibase_core") is None


def test_the_receipt_is_published_through_the_lab_fact_publisher(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "published lab fact (x)\n", "")

    minted = [
        pool.MintedReceipt(
            key="k",
            result="PASS",
            token="ACCEPTED",
            reason="r",
            receipt_path=tmp_path / "r.json",
            event_path=tmp_path / "e.json",
        )
    ]
    lines = pool.publish_events(minted, "dev", tmp_path / "ci_bus_lanes.yaml", runner)
    assert calls[0][1] == str(pool.PUBLISH_LAB_FACT_PY)
    assert calls[0][2:] == [
        "--event",
        str(tmp_path / "e.json"),
        "--bus-lane",
        "dev",
        "--bus-overlay",
        str(tmp_path / "ci_bus_lanes.yaml"),
    ]
    assert lines == ["  publish k: rc=0 published lab fact (x)"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args],
        cwd=repo,
        env=scrub_git_location_env(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_the_host_digest_is_the_verifiers_digest(tmp_path: Path) -> None:
    """pr_subject on the host and compute_pr_diff_digest_from_repo agree byte for byte."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "dev")
    (repo / "a.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-q", "-c", "pr")
    (repo / "a.txt").write_text("head\n", encoding="utf-8")
    (repo / "b.txt").write_text("new\n", encoding="utf-8")
    _git(repo, "add", "a.txt", "b.txt")
    _git(repo, "commit", "-qm", "head")
    head = _git(repo, "rev-parse", "HEAD")
    script = PROVE_SH.read_text(encoding="utf-8")
    start = script.index("pr_subject() {")
    end = script.index("\n}\n", start) + 3
    out = subprocess.run(
        ["bash", "-c", script[start:end] + f'pr_subject "{repo}" r 1 {head} {base}'],
        env=scrub_git_location_env(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (fact,) = pool.pr_subject_facts(out)
    assert (fact.head, fact.base, fact.merge_base) == (head, base, base)
    assert fact.diff_digest == receipts.compute_pr_diff_digest_from_repo(
        repo, base, head
    )


def test_every_receipt_dispatches_the_check_run_workflow_with_itself(
    tmp_path: Path,
) -> None:
    import base64
    import gzip

    code, _, found, _ = _run(tmp_path)
    assert code == pool.EXIT_PASS
    (receipt_path,) = sorted((tmp_path / "receipts").glob("*.receipt.json"))
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    minted = [
        pool.MintedReceipt(
            key="k",
            result="PASS",
            token="ACCEPTED",
            reason="r",
            receipt_path=receipt_path,
            event_path=tmp_path / "e.json",
        )
    ]
    lines = pool.dispatch_check_runs(minted, runner=runner)
    (argv,) = calls
    assert argv[:9] == [
        "gh",
        "workflow",
        "run",
        "lab-proof-receipt.yml",
        "--repo",
        "OmniNode-ai/omnibase_infra",
        "--ref",
        "dev",
        "-f",
    ]
    name, _, value = argv[9].partition("=")
    assert name == "receipt_gzip_b64"
    # the workflow decodes exactly this receipt, byte for byte
    assert gzip.decompress(base64.b64decode(value)) == receipt_path.read_bytes()
    assert json.loads(gzip.decompress(base64.b64decode(value))) == found[0]
    assert lines == ["  check-run dispatch k: rc=0 no output"]


def test_the_check_run_workflow_exists_and_is_dispatch_only() -> None:
    import yaml

    workflow = REPO_ROOT / ".github" / "workflows" / pool.CHECK_RUN_WORKFLOW
    doc = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    triggers = doc[True] if True in doc else doc["on"]
    # dispatched by the pool only; a pull_request trigger would post on heads
    # nobody proved
    assert set(triggers) == {"workflow_dispatch"}
    assert "receipt_gzip_b64" in triggers["workflow_dispatch"]["inputs"]
    text = workflow.read_text(encoding="utf-8")
    assert '--pin "${REPO}" "${KIND}"' in text
    assert "--profile-pin" in text
    assert "--mandatory-check" not in text
