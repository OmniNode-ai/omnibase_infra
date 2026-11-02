# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the pre-push deploy-scope DoD parity gate (OMN-14681).

These tests drive the pure ``classify_deploy_scope`` core against the REAL
canonical deploy-gate validator (imported the same way the hook imports it),
proving both the DRY import wiring and every tri-state outcome -- including the
omnibase_infra#2319 gap (deploy-scoped surface + cited ticket whose contract
declares no deploy-scope DoD evidence).
"""

from __future__ import annotations

import fcntl
import os
import re
import stat
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts.ci import check_deploy_scope_dod as hook
from scripts.ci.check_deploy_scope_dod import (
    FAIL_NO_EVIDENCE,
    FAIL_NO_TICKET,
    PASS_EVIDENCE,
    SKIP_NO_RUNTIME,
    DeployScopeHookError,
    classify_deploy_scope,
    load_canonical_validator,
    resolve_omni_home,
)

# A runtime path that matches the canonical RUNTIME_PATH_PATTERNS
# (src/omnibase_infra/runtime/**/*.py) -- deploy-scoped by construction.
_RUNTIME_FILE = "src/omnibase_infra/runtime/service_kernel.py"
# A non-deploy-scoped path (docs) -- matches no runtime pattern.
_NON_RUNTIME_FILE = (
    "knowledge-base-internal:reference/omnibase-infra-error-handling-patterns.md"
)


@pytest.fixture(scope="module")
def validator():
    """The REAL canonical validator, imported exactly as the hook imports it."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        omni_home = resolve_omni_home(repo_root)
        return load_canonical_validator(omni_home)
    except DeployScopeHookError as exc:
        # Absent OR stale. OMN-16989: the resolver walks every parent of the
        # repo root, so on a transplanted tree (the OMN-16991 remote leg) it can
        # reach an unrelated, months-old omniclaude clone and import it. That
        # produced four reds about another repo's clone freshness with nothing
        # in the message saying so; a named skip is the honest outcome.
        pytest.skip(f"canonical omniclaude validator unusable here: {exc}")


@pytest.fixture(autouse=True)
def _report_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pin the evidence rule to CI's default rollout mode so the test asserts
    # against the same behaviour the hosted gate uses today.
    monkeypatch.setenv("DEPLOY_GATE_FALSIFIABILITY", "report")


def _write_contract(contracts_dir: Path, ticket: str, check_value: str) -> None:
    contracts_dir.mkdir(parents=True, exist_ok=True)
    (contracts_dir / f"{ticket}.yaml").write_text(
        "dod_evidence:\n"
        "  - id: dod-deploy\n"
        "    checks:\n"
        f"      - check_value: {check_value!r}\n",
        encoding="utf-8",
    )


def test_no_runtime_paths_skips(validator, tmp_path: Path) -> None:
    decision = classify_deploy_scope(
        validator=validator,
        changed_files=[_NON_RUNTIME_FILE],
        pr_body="Closes OMN-1234",
        contracts_dir=tmp_path,
    )
    assert decision.outcome == SKIP_NO_RUNTIME
    assert decision.exit_code == 0


def test_runtime_path_without_ticket_fails(validator, tmp_path: Path) -> None:
    decision = classify_deploy_scope(
        validator=validator,
        changed_files=[_RUNTIME_FILE],
        pr_body="a push that cites no ticket at all",
        contracts_dir=tmp_path,
    )
    assert decision.outcome == FAIL_NO_TICKET
    assert decision.exit_code == 1
    assert _RUNTIME_FILE in decision.runtime_hits


def test_runtime_path_ticket_with_deploy_evidence_passes(
    validator, tmp_path: Path
) -> None:
    _write_contract(
        tmp_path,
        "OMN-1234",
        "docker exec ${RUNTIME_CONTAINER:-omninode-runtime} python -c 'import x'",
    )
    decision = classify_deploy_scope(
        validator=validator,
        changed_files=[_RUNTIME_FILE],
        pr_body="Closes OMN-1234",
        contracts_dir=tmp_path,
    )
    assert decision.outcome == PASS_EVIDENCE
    assert decision.exit_code == 0
    assert "OMN-1234" in decision.tickets


def test_runtime_path_ticket_without_deploy_evidence_fails(
    validator, tmp_path: Path
) -> None:
    # The omnibase_infra#2319 gap: contract present, but no deploy-scope probe.
    # Value carries none of the legacy keywords (docker exec / rpk topic
    # produce / deploy) nor a live-surface probe, so it fails in both modes.
    _write_contract(tmp_path, "OMN-1234", "pytest tests/unit -k service_kernel")
    decision = classify_deploy_scope(
        validator=validator,
        changed_files=[_RUNTIME_FILE],
        pr_body="Closes OMN-1234",
        contracts_dir=tmp_path,
    )
    assert decision.outcome == FAIL_NO_EVIDENCE
    assert decision.exit_code == 1
    assert "OMN-1234" in decision.tickets


class _RuntimeAndTicketStandIn(ModuleType):
    """The canonical validator's runtime-path and ticket surface, nothing more."""

    TICKET_PATTERN = re.compile(r"OMN-(\d+)")

    def __init__(self) -> None:
        super().__init__("_stand_in_deploy_gate_validator")

    def find_runtime_paths(self, files: list[str]) -> list[str]:
        return [f for f in files if f == _RUNTIME_FILE]

    def parse_evidence_metadata(self, body: str) -> SimpleNamespace:
        return SimpleNamespace(source="", ticket="")


def test_runtime_path_ticket_without_local_contract_fails(tmp_path: Path) -> None:
    # The hosted gate reads contracts/<ticket>.yaml at the head and fails when it
    # is absent, so the local mirror fails on the same gap. The missing-contract
    # branch is decided before the evidence rule runs, so a stand-in for the
    # canonical validator's runtime and ticket surface keeps this test running on
    # a runner with no omniclaude clone (OMN-18776).
    decision = classify_deploy_scope(
        validator=_RuntimeAndTicketStandIn(),
        changed_files=[_RUNTIME_FILE],
        pr_body="Closes OMN-9999",
        contracts_dir=tmp_path,  # empty -> OMN-9999.yaml absent
    )
    assert decision.outcome == FAIL_NO_EVIDENCE
    assert decision.exit_code == 1
    assert "OMN-9999" in decision.tickets


def test_runtime_detection_is_dry_with_canonical_validator(validator) -> None:
    # Prove the hook's detection is the canonical validator's own function,
    # not a local re-implementation (OMN-14655 DRIFT-3 guard).
    assert validator.find_runtime_paths([_RUNTIME_FILE]) == [_RUNTIME_FILE]
    assert validator.find_runtime_paths([_NON_RUNTIME_FILE]) == []


# =============================================================================
# OMN-16989: a stale sibling clone must fail with its own name
# =============================================================================


def _fake_clone(root: Path, body: str) -> Path:
    """A directory shaped like an OMNI_HOME whose omniclaude clone carries
    BODY as the canonical deploy-gate validator."""
    target = root / "omniclaude" / ".github" / "actions" / "deploy-gate"
    target.mkdir(parents=True)
    (target / "validate_pr_deploy_required.py").write_text(body, encoding="utf-8")
    return root


def test_a_stale_canonical_validator_fails_with_staleness_as_the_named_reason(
    tmp_path: Path,
) -> None:
    """The resolver walks EVERY parent of the repo root, so whichever directory
    named ``omniclaude`` sits nearest above the checkout wins -- with no
    assertion that it is this repo's sibling or anywhere near current.

    Measured on `.201` over the OMN-16991 remote leg: the transplanted tree at
    ``/data/omninode/onex-prepush/runs/<id>/tree`` reached ``/data/omninode``
    and imported an omniclaude clone pinned at omniclaude#1600, four versions of
    the validator behind. Four tests then failed on ``has no attribute
    'parse_evidence_metadata'`` -- a red about another repo's clone freshness,
    on whichever host happened to run the suite, with nothing in the message
    saying so. Silent dependence on a cross-repo hidden input is the defect;
    naming it is the fix."""
    home = _fake_clone(tmp_path, "def find_runtime_paths(files):\n    return []\n")
    with pytest.raises(DeployScopeHookError) as excinfo:
        load_canonical_validator(home)
    message = str(excinfo.value)
    assert "STALE" in message
    assert "parse_evidence_metadata" in message
    assert str(home) in message


def test_a_current_canonical_validator_still_imports(tmp_path: Path) -> None:
    """The staleness assertion must not reject a clone that does expose the
    surface -- it may only add a named failure, never a new way to refuse."""
    home = _fake_clone(
        tmp_path,
        "import re\n"
        "TICKET_PATTERN = re.compile(r'OMN-([0-9]+)')\n"
        "def find_runtime_paths(files):\n    return []\n"
        "def parse_evidence_metadata(body):\n    return None\n",
    )
    module = load_canonical_validator(home)
    assert hasattr(module, "parse_evidence_metadata")


# ---------------------------------------------------------------------------
# Verifier install lock (OMN-21913): no flock(1), one exclusive fcntl lock.
# ---------------------------------------------------------------------------

_TAG = "dev-20261101000200-abcdef012345"
#: Everything of the real install script that runs before it reads the ``installed`` marker.
_LOCK_PREAMBLE = hook._VERIFIER_INSTALL_SCRIPT.split('[[ -f "$2/installed"')[0]


@pytest.fixture
def no_flock_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A PATH whose ``flock`` is the macOS answer: command not found, rc 127."""
    shim = tmp_path / "shim" / "flock"
    shim.parent.mkdir()
    shim.write_text(
        "#!/bin/sh\necho 'flock: command not found' >&2\nexit 127\n", encoding="utf-8"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{shim.parent}:{os.environ['PATH']}")


def _use_timed_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hold_seconds: float
) -> Path:
    """Run the real lock preamble, then a body that logs start and end around a sleep."""
    log = tmp_path / "order.log"
    body = f'echo start >> "{log}"; sleep {hold_seconds}; echo end >> "{log}"'
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(hook, "_VERIFIER_INSTALL_SCRIPT", _LOCK_PREAMBLE + body)
    return log


def test_the_install_script_takes_no_flock_command() -> None:
    """macOS lab hosts ship no flock(1); the bash string must not call it (rc 127)."""
    assert not re.search(r"(?<![.\w])flock\b", hook._VERIFIER_INSTALL_SCRIPT)


@pytest.mark.usefixtures("no_flock_binary")
def test_a_second_install_waits_for_the_first_on_one_lock_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = _use_timed_install(monkeypatch, tmp_path, 0.6)

    threads = [
        threading.Thread(target=hook.install_verifier, args=(_TAG,)) for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert log.read_text(encoding="utf-8").split() == ["start", "end"] * 2


@pytest.mark.usefixtures("no_flock_binary")
def test_an_install_blocks_while_a_peer_holds_the_lock_and_runs_after_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = _use_timed_install(monkeypatch, tmp_path, 0.1)
    lock = tmp_path / "cache" / "onex-dod-verifier" / f"{_TAG}.lock"
    lock.parent.mkdir(parents=True)
    with lock.open("w") as peer:
        fcntl.flock(peer, fcntl.LOCK_EX)
        thread = threading.Thread(target=hook.install_verifier, args=(_TAG,))
        thread.start()
        time.sleep(1.0)
        assert not log.exists(), "the install ran while a peer held the lock"
        fcntl.flock(peer, fcntl.LOCK_UN)
    thread.join(timeout=60)

    assert log.read_text(encoding="utf-8").split() == ["start", "end"]


@pytest.mark.usefixtures("no_flock_binary")
def test_a_lock_path_that_cannot_be_opened_refuses_and_never_installs_unlocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = _use_timed_install(monkeypatch, tmp_path, 0.1)
    (tmp_path / "cache" / "onex-dod-verifier" / f"{_TAG}.lock").mkdir(parents=True)

    with pytest.raises(DeployScopeHookError, match=r"failed \(rc="):
        hook.install_verifier(_TAG)

    assert not log.exists()


@pytest.mark.usefixtures("no_flock_binary")
def test_a_lock_that_cannot_be_taken_refuses_and_never_installs_unlocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = _use_timed_install(monkeypatch, tmp_path, 0.1)
    failing = tmp_path / "failing"
    failing.mkdir()
    (failing / "python3").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    (failing / "python3").chmod(0o755)
    monkeypatch.setenv("PATH", f"{failing}:{os.environ['PATH']}")

    with pytest.raises(DeployScopeHookError, match=r"failed \(rc=1\)"):
        hook.install_verifier(_TAG)

    assert not log.exists()


def test_an_unwritable_lock_directory_fails_the_hook_with_a_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The cache root is a file, so no lock can be created under it, as root too."""
    blocker = tmp_path / "cache"
    blocker.write_text("not a directory\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocker))
    monkeypatch.setattr(hook, "resolve_omni_home", lambda _root: tmp_path)
    monkeypatch.setattr(
        hook,
        "load_canonical_validator",
        lambda _home: SimpleNamespace(DEPLOY_EVIDENCE_GUIDANCE=""),
    )
    monkeypatch.setattr(
        hook,
        "classify_deploy_scope",
        lambda **_kw: SimpleNamespace(
            outcome=SKIP_NO_RUNTIME, exit_code=0, runtime_hits=(), detail=""
        ),
    )
    monkeypatch.setattr(hook, "render", lambda *_a: None)
    monkeypatch.setattr(hook, "read_verifier_tag", lambda _root: _TAG)
    monkeypatch.setattr(hook, "read_head_contract", lambda _root, _ticket: "x: 1\n")

    rc = hook.main(
        [
            "--changed-files",
            "README.md",
            "--pr-body",
            "",
            "--contracts-dir",
            str(tmp_path),
            "--subjects",
            "fix(OMN-21913): subject",
        ]
    )

    assert rc != 0
    err = capsys.readouterr().err
    assert "ERROR (fail-loud): installing omnimarket tag" in err
    assert "Not a directory" in err
