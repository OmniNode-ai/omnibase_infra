# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A pool proof keeps its build log and alarms on repeated INCONCLUSIVE (OMN-19998).

The proof's teardown removes the work directory on the lab host, so the build log
of a failed proof (omnibase_infra#5702 on the pre-PR slot, omnibase_infra#5994 on
the isolated member) was gone before anyone read it, and the proof printed only the
last four lines of it. The driver now reads the logs back over its own transport
before the teardown phase, keeps them beside the receipt and names the failing
build step. Two INCONCLUSIVE proofs in a row for one repo write one alarm row.

The fake host here runs the driver's real read-back command in a real shell
against a real directory; only the teardown phase is the fake host's, and it
removes that directory, as the proof's own teardown does.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.models.lab_proof.model_proof_ledger import ModelProofLedger
from omnibase_infra.models.lab_proof.model_proof_log_bundle import ModelProofLogBundle
from tests.helpers.prepr_pool_overlay import pool_overlay_env

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_PY = REPO_ROOT / "scripts" / "runtime_build" / "prepr_runtime_pool.py"
POOL_TESTS = Path(__file__).with_name("test_prepr_runtime_pool_omn18893.py")
RECEIPT_TESTS = Path(__file__).with_name("test_prepr_pool_pr_head_receipt_omn19566.py")

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("pool_overlay_env")]
_ = pool_overlay_env  # the fixture, registered by import


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pool = _load("prepr_runtime_pool", POOL_PY)
fakes = _load("prepr_runtime_pool_fakes_omn19998", POOL_TESTS)
rcpt = _load("prepr_pool_receipt_tests_omn19998", RECEIPT_TESTS)
CFG = pool.load_pool_config(fakes.POOL_FIXTURE)
NOW = dt.datetime(2026, 11, 15, 20, 0, tzinfo=dt.UTC)

#: a BuildKit plain-progress log of a build whose RUN step failed, long enough
#: that its last four lines name nothing of the step that failed
FAILED_BUILD_LOG = "\n".join(
    [f"#{n} [stage 1/9] RUN step-{n}\n#{n} DONE 0.{n}s" for n in range(1, 8)]
    + [
        "#9 [stage 8/9] RUN make local-env LOCAL_ENV_FILE=/w/local.env",
        "#9 0.412 make: *** [Makefile:88: local-env] Error 2",
        '#9 ERROR: process "/bin/sh -c make local-env" did not complete successfully: exit code: 2',
        "------",
        " > [stage 8/9] RUN make local-env:",
        "------",
        "Dockerfile.runtime:131",
        "ERROR: failed to solve: process did not complete successfully",
        "trailing line one",
        "trailing line two",
        "trailing line three",
        "trailing line four",
    ]
)
SLOT_FAIL_LOG = (
    "[prepr-verify-lane] building the runtime image from the snapshot\n"
    "[prepr-verify-lane] REFUSED/FAILED: the workspace build failed or timed out.\n"
)


class WorkdirTransport(fakes.FakeTransport):
    """The fake host, with a real work directory the teardown phase removes."""

    def __init__(self, hosts: dict[str, Any], home: Path, logs: dict[str, str]) -> None:
        super().__init__(hosts)
        self.home = home
        self.logs = logs
        self.work = home / f"{CFG.work_root_prefix}me-pool-a"
        self.work.mkdir(parents=True)
        for name, text in logs.items():
            (self.work / name).write_text(text, encoding="utf-8")
        (self.work / "local.env").write_text("SECRET=never-kept\n", encoding="utf-8")

    def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
        if "PROOF-LOGS-BEGIN" in command:
            done = subprocess.run(
                ["bash", "-c", command.replace("$HOME", str(self.home))],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                env={"PATH": os.environ["PATH"]},
            )
            return done.returncode, done.stdout + done.stderr
        if command.endswith(" teardown"):
            shutil.rmtree(self.work, ignore_errors=True)
        return super().run(host, command, timeout)


def _prove(
    tmp_path: Path,
    logs: dict[str, str],
    *,
    build: str,
    ledger: list[str] | None = None,
    append: Any = None,
    holder: str = "me",
) -> tuple[int, str, WorkdirTransport, Path]:
    host = fakes.FakeHost()
    host.phases["clone"] = rcpt.CLONE
    host.phases["probe"] = fakes.GOOD["probe"] + rcpt.LIVE
    host.phases["build"] = build
    transport = WorkdirTransport({"pool-a": host}, tmp_path / "home", logs)
    out = tmp_path / "receipts"
    code, text = pool.run_proof(
        CFG,
        transport,
        rcpt._params(tmp_path),
        holder,
        60,
        "pool-a",
        ledger or [],
        now_fn=lambda: NOW,
        log=lambda s: None,
        receipt_dir=out,
        verifier_identity="prepr_runtime_pool.judge@test-mac",
        ledger_append=append,
    )
    return code, text, transport, out


def _kept_dir(text: str) -> Path:
    line = next(x for x in text.splitlines() if "proof-logs kept=" in x)
    return Path(line.split("kept=", 1)[1].split(" files=", 1)[0])


# ---------------------------------------------------------------- the log survives


def test_the_full_build_log_survives_a_teardown_that_removes_the_workdir(
    tmp_path: Path,
) -> None:
    code, text, transport, out = _prove(
        tmp_path, {"build.log": FAILED_BUILD_LOG}, build="build rc=2\nup rc=2"
    )
    assert code == pool.EXIT_INCONCLUSIVE
    assert not transport.work.exists(), "positive control: teardown removed the workdir"
    kept = _kept_dir(text)
    assert kept.parent == out, "the logs sit beside the receipts"
    assert (kept / "build.log").read_text(encoding="utf-8") == FAILED_BUILD_LOG
    # the receipt names that path, so a reader of the receipt finds the log
    receipt = json.loads(next(out.glob("*.receipt.json")).read_text())
    stack = next(c for c in receipt["checks"] if c["name"] == "stack_built")
    assert f"proof-logs kept={kept}" in stack["evidence"]
    # only logs are kept, never the generated env file
    assert [p.name for p in kept.iterdir()] == ["build.log"]


def test_a_failed_build_prints_the_failing_step_not_the_last_four_lines(
    tmp_path: Path,
) -> None:
    _, text, _, out = _prove(
        tmp_path, {"build.log": FAILED_BUILD_LOG}, build="build rc=2\nup rc=2"
    )
    step = next(x for x in text.splitlines() if "build-failing-step" in x)
    assert "step #9" in step and "RUN make local-env" in step
    assert "exit code: 2" in step
    assert "trailing line four" not in step
    receipt = json.loads(next(out.glob("*.receipt.json")).read_text())
    stack = next(c for c in receipt["checks"] if c["name"] == "stack_built")
    assert "build-failing-step" in stack["evidence"]


def test_a_slot_build_log_names_the_entrypoint_failure() -> None:
    bundle = ModelProofLogBundle(files=(("build.log", SLOT_FAIL_LOG),))
    assert "REFUSED/FAILED: the workspace build failed" in bundle.failing_step()


def test_a_passing_build_prints_no_failing_step(tmp_path: Path) -> None:
    code, text, _, _ = _prove(
        tmp_path,
        {"build.log": "#1 [1/1] RUN true\n#1 DONE 0.1s\n"},
        build=fakes.GOOD["build"],
    )
    assert code == pool.EXIT_PASS, text
    assert "proof-logs kept=" in text and "build-failing-step" not in text


def test_logs_already_gone_are_reported_not_invented(tmp_path: Path) -> None:
    host = fakes.FakeHost()
    host.phases["clone"] = rcpt.CLONE
    transport = WorkdirTransport({"pool-a": host}, tmp_path / "home", {})
    shutil.rmtree(transport.work)
    _, text = pool.run_proof(
        CFG,
        transport,
        rcpt._params(tmp_path),
        "me",
        60,
        "pool-a",
        [],
        now_fn=lambda: NOW,
        log=lambda s: None,
        receipt_dir=tmp_path / "receipts",
        verifier_identity="prepr_runtime_pool.judge@test-mac",
    )
    assert "proof-logs not kept" in text


def test_an_archive_with_a_path_traversal_member_keeps_nothing() -> None:
    import base64
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("../../escape.sh")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"x=1"))
    text = f"PROOF-LOGS-BEGIN\n{base64.b64encode(buffer.getvalue()).decode()}\nPROOF-LOGS-END"
    assert ModelProofLogBundle.from_archive(text).files == ()


# --------------------------------------------------------------------- the alarm

PROOF_ONE = "2026-11-14T19:27:38Z-landing-lab-omnibase_infra-L559"


def _release(proof_id: str, verdict: str, subject: str) -> str:
    result = "PASS" if verdict == "PASS" else "ABORTED"
    return (
        f"2026-11-14T19:34:46Z | RELEASE | lane=landing-lab-omnibase_infra-L559 | "
        f"re={proof_id} | ticket=OMN-18893 | surface=a-runtime | result={result} | "
        f"LAB PROOF {verdict}: {subject} on pool-a, "
        "isolated project omnibase-infra-local"
    )


def _inconclusive_run(tmp_path: Path, ledger: list[str]) -> tuple[str, list[str]]:
    rows: list[str] = []
    _, text, _, _ = _prove(
        tmp_path,
        {"build.log": FAILED_BUILD_LOG},
        build="build rc=2\nup rc=2",
        ledger=ledger,
        append=rows.append,
    )
    return text, rows


def test_two_inconclusive_proofs_in_a_row_for_one_repo_write_the_alarm_row(
    tmp_path: Path,
) -> None:
    # the fixture proof is of omnimarket#3042: the earlier proof was INCONCLUSIVE too
    ledger = [_release(PROOF_ONE, "INCONCLUSIVE", "omnimarket#3042 head c050011081")]
    text, rows = _inconclusive_run(tmp_path, ledger)
    assert len(rows) == 1, text
    row = rows[0]
    assert " | STATUS | lane=lab-proof-alarm | " in row
    assert "event=lab-proof-inconclusive-streak" in row and "repo=omnimarket" in row
    assert f"proofs={PROOF_ONE},2026-11-15T20:00:00Z-me" in row
    assert "alarm written: lab-proof-inconclusive-streak repo=omnimarket" in text


def test_an_inconclusive_proof_after_a_pass_writes_no_alarm(tmp_path: Path) -> None:
    ledger = [
        _release("old-one", "INCONCLUSIVE", "omnimarket#3000 head aaaaaaaaaa"),
        _release(PROOF_ONE, "PASS", "omnimarket#3042 head c050011081"),
    ]
    text, rows = _inconclusive_run(tmp_path, ledger)
    assert rows == [] and "alarm" not in text


def test_two_inconclusive_proofs_for_different_repos_write_no_alarm(
    tmp_path: Path,
) -> None:
    ledger = [
        _release(PROOF_ONE, "INCONCLUSIVE", "omnibase_infra#5702 head c050011081")
    ]
    text, rows = _inconclusive_run(tmp_path, ledger)
    assert rows == [] and "alarm" not in text


def test_a_failed_append_is_reported_not_swallowed(tmp_path: Path) -> None:
    def refuse(row: str) -> None:
        raise RuntimeError("onex-ledger exited 65: refused")

    ledger = [_release(PROOF_ONE, "INCONCLUSIVE", "omnimarket#3042 head c050011081")]
    _, text, _, _ = _prove(
        tmp_path,
        {"build.log": FAILED_BUILD_LOG},
        build="build rc=2\nup rc=2",
        ledger=ledger,
        append=refuse,
    )
    assert "alarm NOT written (onex-ledger exited 65: refused)" in text


def test_no_configured_append_says_so_instead_of_staying_quiet() -> None:
    ledger = ModelProofLedger(
        lines=(_release(PROOF_ONE, "INCONCLUSIVE", "omnimarket#3042 head c050011081"),)
    )
    note = ledger.alarm_on_streak(
        ["omnimarket"], "now-me", "2026-11-15T20:00:00Z", ModelProofLedger.no_append
    )
    assert "alarm NOT written (no ledger append configured)" in note


def test_the_newest_proof_of_a_repo_decides_not_an_older_one() -> None:
    ledger = ModelProofLedger(
        lines=(
            _release("a", "INCONCLUSIVE", "omnimarket#1 head x"),
            _release("b", "INCONCLUSIVE", "omnimarket#2 head y"),
            _release("c", "PASS", "omnimarket#3 head z"),
        )
    )
    assert ledger.last_proof_verdicts()["omnimarket"] == ("PASS", "c")
    rows: list[str] = []
    assert ledger.alarm_on_streak(["omnimarket"], "d", "t", rows.append) == ""
    assert rows == []


def test_the_alarm_row_is_one_status_row_the_ledger_grammar_reads() -> None:
    row = ModelProofLedger.streak_row(
        "omnimarket", "earlier", "current", "2026-11-15T20:00:00Z"
    )
    cells = [c.strip() for c in row.split(" | ")]
    assert cells[1] == "STATUS" and cells[2] == "lane=lab-proof-alarm"
    assert "\n" not in row
    assert all("|" not in c for c in cells)


def test_the_live_hold_reader_moved_out_of_the_driver_still_answers() -> None:
    lines = [
        "2026-11-15T19:00:00Z | HOLD | lane=a | id=h1 | surface=202-runtime | "
        "until=2026-11-15T21:00:00Z | x",
        "2026-11-15T19:01:00Z | HOLD | lane=b | id=h2 | surface=a-runtime | "
        "until=2026-11-15T21:00:00Z | x",
        "2026-11-15T19:02:00Z | RELEASE | lane=b | re=h2 | surface=a-runtime | x",
    ]
    held = pool.live_surface_holds(lines, NOW)
    assert held == {"202-runtime": ("h1", "a", "2026-11-15T21:00:00Z")}


def test_the_appender_passes_the_row_as_one_argument(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    fake = tmp_path / "fake-ledger.py"
    fake.write_text(
        "import sys, pathlib\n"
        "argv = sys.argv[1:]\n"
        "assert argv[1:3] == ['--timeout', '60s'] and argv[3] == '--append', argv\n"
        "pathlib.Path(argv[0]).open('a').write(argv[4] + '\\n')\n",
        encoding="utf-8",
    )
    append = ModelProofLedger.appender(f"{sys.executable} {fake}", str(ledger))
    append("2026-11-15T20:00:00Z | STATUS | lane=x | a row; with 'quotes' and $vars")
    assert (
        ledger.read_text(encoding="utf-8").strip().endswith("with 'quotes' and $vars")
    )
    assert ModelProofLedger.appender("", str(ledger)) is ModelProofLedger.no_append
    assert ModelProofLedger.appender("x", None) is ModelProofLedger.no_append
