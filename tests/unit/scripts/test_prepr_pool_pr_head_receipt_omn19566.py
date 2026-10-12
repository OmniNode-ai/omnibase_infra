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
CFG = pool.load_pool_config(fakes.POOL_FIXTURE)
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
        fakes.FakeTransport({"pool-a": host}),
        _params(tmp_path),
        "plan-item-I7",
        60,
        "pool-a",
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
    assert subject.profile_version == "2"
    assert subject.handler_kind is receipts.EnumLabProofHandlerKind.RUNTIME_IMAGE
    assert subject.runner_identity == "plan-item-I7@pool-a"
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
        f"receipt OmniNode-ai/omnimarket#3042@{HEAD}:omnimarket.runtime_image@2 "
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
    _, text, _, events = _run(tmp_path, verifier="PLAN-ITEM-I7@pool-a")
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
    assert subject["host"] == "pool-a"
    assert CFG.host("pool-a").ssh_target not in json.dumps(found[0])


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
        fakes.FakeTransport({"pool-a": host}),
        _params(tmp_path),
        "plan-item-I7",
        60,
        "pool-a",
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


def test_a_run_dispatches_the_check_run_unless_told_not_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # OMN-19566 slice 3 was built opt-in and nothing passed the flag, so no
    # pr-head receipt ever reached lab-proof-receipt.yml. The default is to post.
    dispatched: list[int] = []
    monkeypatch.setattr(
        pool, "dispatch_check_runs", lambda minted: dispatched.append(len(minted)) or []
    )
    default = pool.build_publisher(None, None, dispatch_check_run=True)
    assert default is not None
    default([object(), object()])
    assert dispatched == [2]
    assert pool.build_publisher(None, None, dispatch_check_run=False) is None


def test_the_run_parser_posts_by_default_and_has_one_opt_out() -> None:
    args = pool.build_parser().parse_args(["run", "--params", "p.env", "--holder", "h"])
    assert args.no_dispatch_check_run is False
    off = pool.build_parser().parse_args(
        ["run", "--params", "p.env", "--holder", "h", "--no-dispatch-check-run"]
    )
    assert off.no_dispatch_check_run is True
    with pytest.raises(SystemExit):
        pool.build_parser().parse_args(
            ["run", "--params", "p.env", "--holder", "h", "--dispatch-check-run"]
        )


@pytest.mark.parametrize(
    "error", [FileNotFoundError("gh"), subprocess.TimeoutExpired("gh", 60)]
)
def test_a_dispatch_that_cannot_run_is_reported_not_raised(
    tmp_path: Path, error: Exception
) -> None:
    receipt = tmp_path / "r.json"
    receipt.write_text("{}", encoding="utf-8")
    minted = [
        pool.MintedReceipt(
            key="k",
            result="PASS",
            token="ACCEPTED",
            reason="r",
            receipt_path=receipt,
            event_path=tmp_path / "e.json",
        )
    ]

    def runner(*_a: Any, **_k: Any) -> Any:
        raise error

    (line,) = pool.dispatch_check_runs(minted, runner=runner)
    assert line.startswith("  check-run dispatch k: not dispatched")


# --- conditional checks resolved from the PR's own diff (OMN-19566) ---------

INFRA_CLONE = (
    f"omnibase_infra#4809 fetched head {HEAD} expected {HEAD} match=yes\n"
    f"pr-subject omnibase_infra#4809 head={HEAD} base={DEV} merge-base={MERGE_BASE} "
    f"diff-digest={DIGEST}\n"
    "{changed}"
    f"omnibase_infra test-merge clean {'6' * 40} (dev {DEV})\n"
    "clone done"
)
PIN_ADVANCE = (
    ".github/omnimarket-contract-pin.yaml",
    ".github/sibling-pins.yaml",
    "docker/catalog/database-topology/omninode_internal.yaml",
    "tests/ci/test_supplemental_declaration_expiry_omn18863.py",
)
NODE = "src/omnibase_infra/nodes/node_runtime_error_triage_effect/handlers/h.py"
FORWARDER_CONTRACT = "src/omnibase_infra/nodes/node_bus_forwarder_effect/contract.yaml"


def _infra_clone(*changed: str) -> str:
    return INFRA_CLONE.replace(
        "{changed}", "".join(f"pr-changed omnibase_infra#4809 {p}\n" for p in changed)
    )


def _infra_run(
    tmp_path: Path, clone: str, live: str = ""
) -> tuple[int, str, list[dict[str, Any]]]:
    host = fakes.FakeHost()
    host.phases["clone"] = clone
    host.phases["probe"] = fakes.GOOD["probe"] + live
    params = tmp_path / "p.env"
    params.write_text(f"INFRA_PR=4809\nINFRA_HEAD={HEAD}\n", encoding="utf-8")
    out = tmp_path / "receipts"
    code, text = pool.run_proof(
        CFG,
        fakes.FakeTransport({"pool-a": host}),
        params,
        "labproof-gate",
        60,
        "pool-a",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
        receipt_dir=out,
        verifier_identity="prepr_runtime_pool.judge@test-mac",
    )
    events = [json.loads(p.read_text()) for p in sorted(out.glob("*.event.json"))]
    return code, text, events


def test_a_pin_advance_proved_green_is_accepted_without_the_surface_checks(
    tmp_path: Path,
) -> None:
    code, text, events = _infra_run(tmp_path, _infra_clone(*PIN_ADVANCE))
    assert code == pool.EXIT_PASS, text
    assert events[0]["verifier_token"] == "ACCEPTED", text
    assert "changed_path_live" not in events[0]["mandatory_checks"]
    assert "forwarder_refused_topic" not in events[0]["mandatory_checks"]
    assert "touch no surface of conditional check(s)" in text
    assert ":omnibase_infra.runtime_and_scripts@2 result=PASS verifier=ACCEPTED" in text


def test_the_same_pass_without_its_changed_paths_waives_nothing(
    tmp_path: Path,
) -> None:
    _, text, events = _infra_run(tmp_path, _infra_clone())
    assert events[0]["verifier_token"] == "MISSING_MANDATORY_CHECK"
    assert events[0]["missing_mandatory_checks"] == [
        "changed_path_live",
        "forwarder_refused_topic",
    ]
    assert "no changed paths in the clone output" in text


def test_a_node_change_with_no_live_line_is_missing_changed_path_live(
    tmp_path: Path,
) -> None:
    _, _, events = _infra_run(tmp_path, _infra_clone(NODE))
    assert events[0]["verifier_token"] == "MISSING_MANDATORY_CHECK"
    assert events[0]["missing_mandatory_checks"] == ["changed_path_live"]


def test_a_node_change_whose_derived_term_is_live_is_accepted(
    tmp_path: Path,
) -> None:
    term = pool.LIVE_TERM_DOT + "node_runtime_error_triage_effect" + pool.LIVE_TERM_DOT
    live = (
        f"live-grep derived from the diff: {term} \n"
        f"  omnibase-infra-local-omninode-runtime mentions {term}: 4\n"
        f"  omnibase-infra-local-runtime-effects mentions {term}: 0\n"
    )
    code, text, events = _infra_run(tmp_path, _infra_clone(NODE), live)
    assert code == pool.EXIT_PASS, text
    assert events[0]["verifier_token"] == "ACCEPTED", text
    assert "changed_path_live" in events[0]["mandatory_checks"]


def test_a_node_change_whose_derived_term_is_dead_fails_naming_the_node(
    tmp_path: Path,
) -> None:
    term = pool.LIVE_TERM_DOT + "node_runtime_error_triage_effect" + pool.LIVE_TERM_DOT
    live = (
        f"  omnibase-infra-local-omninode-runtime mentions {term}: 0\n"
        f"  omnibase-infra-local-runtime-effects mentions {term}: 0\n"
    )
    code, text, events = _infra_run(tmp_path, _infra_clone(NODE), live)
    assert code == pool.EXIT_FAIL
    assert (
        "changed path not live, no runtime log mentions: "
        "node_runtime_error_triage_effect\n"
    ) in text + "\n"
    assert events[0]["failing_checks"] == ["changed_path_live"]


def test_a_forwarder_change_stays_missing_the_check_no_pool_stack_can_run(
    tmp_path: Path,
) -> None:
    term = pool.LIVE_TERM_DOT + "node_bus_forwarder_effect" + pool.LIVE_TERM_DOT
    live = f"  omnibase-infra-local-runtime-effects mentions {term}: 2\n"
    _, _, events = _infra_run(tmp_path, _infra_clone(FORWARDER_CONTRACT), live)
    assert events[0]["verifier_token"] == "MISSING_MANDATORY_CHECK"
    assert events[0]["missing_mandatory_checks"] == ["forwarder_refused_topic"]


def test_pr_changed_lines_are_read_per_pull_request() -> None:
    clone = (
        "pr-changed omnibase_infra#1 a/b.py\n"
        "pr-changed omnibase_infra#2 c d/e.py\n"
        "pr-changed omnibase_infra#1 f.yaml\n"
    )
    assert pool.pr_changed_paths(clone) == {
        ("omnibase_infra", 1): ["a/b.py", "f.yaml"],
        ("omnibase_infra", 2): ["c d/e.py"],
    }


def _shell_function(name: str) -> str:
    script = PROVE_SH.read_text(encoding="utf-8")
    start = script.index(f"{name}() {{")
    end = script.index("\n}\n", start) + 3
    return script[start:end]


def test_the_host_prints_the_changed_paths_the_workflow_reads(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "dev")
    (repo / "a.txt").write_text("base\n", encoding="utf-8")
    (repo / "old.txt").write_text("moved\n", encoding="utf-8")
    _git(repo, "add", "a.txt", "old.txt")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-q", "-c", "pr")
    (repo / "a.txt").write_text("head\n", encoding="utf-8")
    _git(repo, "mv", "old.txt", "new.txt")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "head")
    head = _git(repo, "rev-parse", "HEAD")
    out = subprocess.run(
        [
            "bash",
            "-c",
            _shell_function("pr_subject") + f'pr_subject "{repo}" r 7 {head} {base}',
        ],
        env=scrub_git_location_env(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    # the workflow's own command (lab-proof-receipt.yml), over the same range
    workflow = _git(repo, "diff", "--name-only", "--no-renames", base, head)
    assert pool.pr_changed_paths(out) == {("r", 7): workflow.splitlines()}
    assert sorted(workflow.splitlines()) == ["a.txt", "new.txt", "old.txt"]


def test_live_terms_are_derived_from_changed_nodes_that_exist_at_the_head(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    repo = root / "omnibase_infra"
    nodes = repo / "src" / "omnibase_infra" / "nodes"
    for node in ("node_kept", "node_gone", "node_untouched"):
        (nodes / node).mkdir(parents=True)
        (nodes / node / "contract.yaml").write_text("name: x\n", encoding="utf-8")
    (nodes / "__init__.py").write_text("", encoding="utf-8")
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    _git(repo, "init", "-q", "-b", "dev")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "update-ref", "refs/remotes/origin/dev", "HEAD")
    _git(repo, "switch", "-q", "-c", "pr")
    (nodes / "node_kept" / "handler.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "rm", "-rq", str(nodes / "node_gone"))
    (nodes / "__init__.py").write_text("# touched\n", encoding="utf-8")
    (repo / "README.md").write_text("y\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "head")
    script = _shell_function("derive_live_grep")
    out = subprocess.run(
        ["bash", "-c", f'R="{root}"\n' + script + "derive_live_grep omnibase_infra"],
        env=scrub_git_location_env(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert out.split() == [pool.LIVE_TERM_DOT + "node_kept" + pool.LIVE_TERM_DOT]
    # live means the node ran (its consumer group, its own logger), never that
    # it was discovered; and node_kept is not live on node_kept_more's lines
    for line, expected in (
        ("Discovered contract: node_kept (EFFECT_GENERIC) from omnibase_infra", "0"),
        ("contract=node_kept, skipping Kafka subscription for 'node_kept'", "0"),
        ("consumer_group=local.omnibase_infra.node_kept_more.consume.1.0.0", "0"),
        ("consumer_group=local.omnibase_infra.node_kept.consume.1.0.0", "1"),
        ("[INFO] omnibase_infra.nodes.node_kept.handlers.handler_x: ran", "1"),
    ):
        count = subprocess.run(
            ["grep", "-c", out.split()[0]],
            input=line + "\n",
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        assert count == expected
