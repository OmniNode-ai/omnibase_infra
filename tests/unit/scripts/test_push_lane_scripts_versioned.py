# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The .201 push-lane queue scripts are versioned here, not only on the host.

OMN-17392, closing OMN-17221 DoD4. Until this landed, the entire governed `.201`
pre-push queue -- a durable FIFO runner with fcntl locking, an atomic journal,
owner-only mode enforcement, and a YAML contract validator -- existed ONLY at
``~/push-lanes/`` on one machine. No history, no review, no backup. Two separate
tickets (OMN-16968, OMN-17221) had to reconstruct its behavior by SSHing in and
reading it, and OMN-17221's original report got the mechanism WRONG as a result
(it blamed a pgrep pattern that a fresh probe then showed was correct).

These tests are deliberately CHEAP and OFFLINE. They cannot reach `.201` from
CI, so they do not pretend to: they pin that the code is here, parses, and keeps
the invariants that make the queue trustworthy. Host/versioned drift is the
deploy script's job (it verifies sha256 after install).
"""

from __future__ import annotations

import ast
import json
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
LANES = REPO_ROOT / "scripts" / "push_lanes"
DEPLOY = LANES / "deploy-push-lanes.sh"

#: Every artifact that must be versioned. Adding a file to the host without
#: adding it here is exactly the drift this ticket closes.
EXPECTED = {
    "queue-runner.py",
    "queue-runner.sh",
    "queue-contract-validator.py",
    "detect_foreign_prepush.py",
    "README-QUEUE.md",
    "deploy-push-lanes.sh",
    "DEPLOYED_SHA256.json",
}


def test_every_expected_artifact_is_versioned() -> None:
    assert LANES.is_dir(), f"expected the versioned queue scripts at {LANES}"
    assert {p.name for p in LANES.iterdir()} == EXPECTED


@pytest.mark.parametrize(
    "name",
    ["queue-runner.py", "queue-contract-validator.py", "detect_foreign_prepush.py"],
)
def test_the_python_runners_parse(name: str) -> None:
    """A syntax error here bricks the whole .201 heavy-suite lane, and the only
    place that would previously have surfaced is a wedged queue on the host."""
    ast.parse((LANES / name).read_text(encoding="utf-8"), filename=name)


def test_the_bash_entry_point_execs_the_python_runner() -> None:
    """`queue-runner.sh` is a compatibility shim: the implementation moved to
    Python but every lane, runbook and log line still names the `.sh`. If the
    shim stops pointing at the real runner, lanes fail with a shell error that
    names the wrong file."""
    text = (LANES / "queue-runner.sh").read_text(encoding="utf-8")
    assert "queue-runner.py" in text
    assert "exec " in text
    subprocess.run(
        ["bash", "-n", str(LANES / "queue-runner.sh")], check=True, capture_output=True
    )


def test_the_deploy_script_covers_every_versioned_artifact() -> None:
    """The failure mode this prevents: someone versions a new helper here, the
    deploy script never learns about it, and the host silently keeps running the
    old one -- which is the same class of drift as not versioning it at all."""
    deploy = DEPLOY.read_text(encoding="utf-8")
    shipped = EXPECTED - {"deploy-push-lanes.sh", "DEPLOYED_SHA256.json"}
    for name in shipped:
        assert f'"{name}:' in deploy, f"{name} is versioned but never deployed"


def test_the_deploy_script_verifies_what_it_installed() -> None:
    """A deploy that does not read back is a claim, not evidence."""
    deploy = DEPLOY.read_text(encoding="utf-8")
    assert "sha256sum" in deploy
    assert "shasum -a 256" in deploy
    assert "MISMATCH" in deploy
    subprocess.run(["bash", "-n", str(DEPLOY)], check=True, capture_output=True)


def test_the_deploy_script_never_touches_queue_state() -> None:
    """Code is deployable; queue STATE is not. Mutating QUEUE/journal/lanes from
    a deploy would break in-flight lanes and violates the OMN-17221 constraint
    that scoping and fix work must not disturb queued or running work."""
    deploy = DEPLOY.read_text(encoding="utf-8")
    for forbidden in ("QUEUE.journal", "rm -rf", "pkill", "kill -"):
        assert forbidden not in deploy, f"deploy script touches {forbidden}"
    # It may NAME the queue in prose, but must never write it.
    assert "> ${DEST}/QUEUE" not in deploy
    assert "mv ${DEST}/QUEUE" not in deploy


def test_the_deploy_script_preserves_owner_only_modes() -> None:
    """queue-runner.py REFUSES to run against a queue whose parent dir or
    artifacts are group/world accessible (it checks st_uid and S_IMODE). A
    deploy that widened modes would not merely loosen security, it would stop
    the runner outright -- so the modes are pinned next to the runner's own
    check rather than left to whatever umask the deploying shell had."""
    deploy = DEPLOY.read_text(encoding="utf-8")
    assert "chmod ${mode}" in deploy
    assert '"queue-runner.py:700"' in deploy
    runner = (LANES / "queue-runner.py").read_text(encoding="utf-8")
    assert "S_IMODE" in runner and "st_uid" in runner


def test_the_runner_keeps_its_fail_closed_queue_invariants() -> None:
    """The properties that make this queue trustworthy, pinned so a future edit
    has to break a test rather than only a habit: exclusive flock, atomic
    replace + fsync, a rejected-by-schema journal, and a lane-name regex that
    refuses control bytes and path traversal."""
    runner = (LANES / "queue-runner.py").read_text(encoding="utf-8")
    assert "fcntl.flock" in runner and "LOCK_EX" in runner
    assert "os.replace" in runner and "os.fsync" in runner
    assert "LANE_RE" in runner
    assert "journal schema rejected" in runner
    assert "queue contains a raw control byte" in runner


def test_the_recorded_deployed_hashes_are_wellformed() -> None:
    """DEPLOYED_SHA256.json records what was on `.201` when these files were
    lifted, so the header-only delta this repo added is auditable rather than
    asserted."""
    data = json.loads((LANES / "DEPLOYED_SHA256.json").read_text(encoding="utf-8"))
    assert "_README" in data, (
        "the manifest must SAY what the recorded hashes mean and how the "
        "versioned copies differ from them; a bare digest list is not auditable"
    )
    digests = {k: v for k, v in data.items() if not k.startswith("_")}
    assert set(digests) == {
        "queue-runner.py",
        "queue-runner.sh",
        "queue-contract-validator.py",
        "detect_foreign_prepush.py",
    }
    for name, digest in digests.items():
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), name


def test_the_manifest_admits_the_host_is_not_yet_redeployed() -> None:
    """Honesty pin. The versioned copies are NOT byte-identical to `.201` right
    now -- they carry an SPDX header and a `NoReturn` import fix. Claiming
    otherwise would make this directory look like proven host truth when it is
    so far only reviewed truth, which is precisely the kind of unearned claim
    that made the unversioned queue hard to reason about in the first place."""
    readme = json.loads((LANES / "DEPLOYED_SHA256.json").read_text(encoding="utf-8"))[
        "_README"
    ]
    assert "has NOT been redeployed" in readme
    assert "deploy-push-lanes.sh" in readme


def test_the_shipped_scripts_are_executable() -> None:
    for name in ("queue-runner.sh", "queue-runner.py", "deploy-push-lanes.sh"):
        mode = (LANES / name).stat().st_mode
        assert mode & stat.S_IXUSR, f"{name} is versioned without its exec bit"
