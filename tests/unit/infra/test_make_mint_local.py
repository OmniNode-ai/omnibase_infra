# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20823: ``make mint-local`` against recording stubs.

The real targets run. ``docker`` and the host's ``onex`` are stubs on ``PATH``
backed by one fake broker (a JSON file): the stub ``onex node`` inside the fake
``runtime-effects`` container reads the input file the target copied in and
answers as the orchestrator would, and every password the Admin API would set
is recorded as one line of a mutation journal. ``jq``, ``openssl``, ``id`` and
``hostname`` are the host's own.

What the tests read:
* AC2/AC3: create, then unchanged with no mutation, the drift refusal with the
  extra ACL named and the password untouched, and rotate after a logout;
* AC4 (``-k lock``): two runs started together make one mutation and one prints
  ``mint-local: busy`` naming the holder with a non-zero status; a lock whose
  recorded process is dead is removed and said; a live one is not;
* AC5: the password is in no argument of any stub call, in no output, and not
  in the input file, which is gone from the container after the run;
* AC7 (``-k tenant``): the probe carries the stack's tenant from an overridden
  ``LOCAL_ENV_FILE`` while the shell exports another and the default file names
  a third;
* AC8 (``-k no_terminal``): ``verified`` never prints when the probe wrote no
  correlated terminal event.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        shutil.which("make") is None or shutil.which("jq") is None,
        reason="make mint-local needs make and jq",
    ),
]

_REPO = Path(__file__).resolve().parents[3]
_SUPERUSER_PASSWORD = "stub-superuser-secret"

# The fake container and broker. Python, so the stubs share one implementation
# of the broker state; every call is journalled as its argv, one JSON line.
_STUB_DOCKER = r"""#!__PYTHON__
import json, os, re, subprocess, sys, time
from pathlib import Path

argv = sys.argv[1:]
root = Path(os.environ["STUB_ROOT"])
with open(root / "calls.jsonl", "a") as journal:
    journal.write(json.dumps(["docker", *argv]) + "\n")
fs = root / "container"
broker_file = root / "broker.json"

def broker():
    return json.loads(broker_file.read_text())

def save(state):
    broker_file.write_text(json.dumps(state))

if argv[:1] == ["info"]:
    sys.exit(0)
if argv[:1] == ["cp"]:
    src, dst = argv[1], argv[2].split(":", 1)[1]
    target = fs / dst.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(src).read_bytes())
    sys.exit(0)
if argv[:1] != ["exec"]:
    print(f"stub docker: unexpected {argv}", file=sys.stderr)
    sys.exit(2)
args = argv[1:]
inner = {}
while args and args[0] in ("-u", "-e"):
    flag, value = args[0], args[1]
    args = args[2:]
    if flag == "-e":
        name, _, given = value.partition("=")
        if given or "=" in value:
            inner[name] = given
        elif name in os.environ:
            inner[name] = os.environ[name]
args = args[1:]  # the container name
cmd = args

if cmd[:3] == ["rpk", "cluster", "info"]:
    user = inner.get("RPK_USER", "")
    if user and inner.get("RPK_PASS") == os.environ["STUB_SUPERUSER_PASSWORD"]:
        sys.exit(0)
    if os.environ.get("STUB_BROKER_OPEN") == "1":
        sys.exit(0)
    print("unable to request metadata: SASL is required but not provided", file=sys.stderr)
    sys.exit(1)
if cmd[:2] == ["mkdir", "-p"]:
    (fs / cmd[2].lstrip("/")).mkdir(parents=True, exist_ok=True)
    sys.exit(0)
if cmd[:2] == ["rm", "-rf"]:
    subprocess.run(["rm", "-rf", str(fs / cmd[2].lstrip("/"))], check=False)
    sys.exit(0)
if cmd[:2] == ["sh", "-c"] and "grep" in cmd[2]:
    secret = inner.get("ONEX_MINT_PASSWORD", "")
    target = fs / cmd[4].lstrip("/")
    found = subprocess.run(["grep", "-rqF", "-e", secret, str(target)]).returncode
    sys.exit(found)
if cmd[:2] == ["onex", "node"] and cmd[2] == "node_developer_principal_mint_orchestrator":
    opts = dict(zip(cmd[3::2], cmd[4::2]))
    request = json.loads((fs / opts["--input"].lstrip("/")).read_text())
    state_root = fs / opts["--state-root"].lstrip("/")
    state_root.mkdir(parents=True, exist_ok=True)
    time.sleep(float(os.environ.get("STUB_NODE_DELAY", "0")))
    slug = lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    principal = f"dev-{slug(request['login'])}-{slug(request['device'])}"
    state = broker()
    users = state.setdefault("users", {})
    extras = state.get("extra_acls", {}).get(principal, [])
    exists = principal in users
    result = {"mode": request["mode"], "principal": principal, "exists": exists,
              "extra_acls": extras, "missing_acls": [] if exists else ["planned"],
              "status": "success", "action": "audited", "failed_step": "", "reason": ""}
    if request["mode"] == "converge":
        secret = inner.get(request["password_env"], "")
        if not secret:
            result.update(status="failure", failed_step="apply", reason="password variable unset", action="")
        elif extras:
            result.update(status="failure", action="refused", failed_step="apply",
                          reason=f"{principal} holds ACLs outside the plan")
        else:
            users[principal] = secret
            state.setdefault("acls", {})[principal] = ["planned"]
            save(state)
            with open(root / "mutations", "a") as m:
                m.write(f"set_password {principal} caller={os.getppid()}\n")
            result.update(action="rotate" if exists else "create", verified=True, missing_acls=[])
            if os.environ.get("STUB_LEAK_TO_STATE") == "1":
                (state_root / "leak.txt").write_text(secret)
    (state_root / "result.json").write_text(json.dumps(result))
    # The shape `onex node --output receipt` prints, read from a real run 2026-10-30.
    print(json.dumps({"status": "success" if result["status"] == "success" else "failed",
                      "result": {"handler_result": result, "terminal_payload": result}}))
    sys.exit(0 if result["status"] == "success" else 1)
print(f"stub docker: unexpected exec {cmd}", file=sys.stderr)
sys.exit(2)
"""

_STUB_ONEX = r"""#!__PYTHON__
import json, os, sys, uuid
from pathlib import Path

argv = sys.argv[1:]
root = Path(os.environ["STUB_ROOT"])
with open(root / "calls.jsonl", "a") as journal:
    journal.write(json.dumps(["onex", *argv, {"ONEX_TENANT_ID": os.environ.get("ONEX_TENANT_ID")}]) + "\n")
store = root / "store.json"
if argv[:2] == ["auth", "lane-login"]:
    opts = dict(zip(argv[2::2], argv[3::2]))
    password = sys.stdin.readline().rstrip("\n")
    store.write_text(json.dumps({"lane": opts["--lane"], "username": opts["--sasl-username"], "password": password}))
    print(f"Stored bus identity '{opts['--sasl-username']}' for lane '{opts['--lane']}'.")
    sys.exit(0)
if argv[:1] == ["delegate"]:
    state_root = Path(argv[argv.index("--state-root") + 1])
    if not store.exists():
        print("lane 'local' declares SASL_PLAINTEXT and this machine holds no identity for it. Store one with 'onex auth lane-login --lane local ...'", file=sys.stderr)
        sys.exit(1)
    held = json.loads(store.read_text())
    users = json.loads((root / "broker.json").read_text()).get("users", {})
    if users.get(held["username"]) != held["password"]:
        print("SASL authentication failed", file=sys.stderr)
        sys.exit(1)
    run_id, cid = str(uuid.uuid4()), str(uuid.uuid4())
    if os.environ.get("STUB_NO_TERMINAL") != "1":
        run = state_root / "runs" / run_id
        run.mkdir(parents=True)
        (run / "workflow_result.json").write_text(json.dumps(
            {"terminal_payload": {"correlation_id": cid, "tenant_id": os.environ.get("ONEX_TENANT_ID")}}))
    print(json.dumps({"status": "success", "run_id": run_id, "correlation_id": cid}))
    sys.exit(0)
print(f"stub onex: unexpected {argv}", file=sys.stderr)
sys.exit(2)
"""


class _Stack:
    """One fake laptop stack: stubs on PATH, a broker file and an env file."""

    def __init__(self, tmp_path: Path, *, env_text: str | None = None) -> None:
        self.root = tmp_path / "stub"
        self.root.mkdir()
        (self.root / "container").mkdir()
        (self.root / "broker.json").write_text("{}")
        (self.root / "mutations").write_text("")
        (self.root / "calls.jsonl").write_text("")
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        for name, body in (("docker", _STUB_DOCKER), ("onex", _STUB_ONEX)):
            stub = self.bin / name
            stub.write_text(
                body.replace("__PYTHON__", sys.executable), encoding="utf-8"
            )
            stub.chmod(0o755)
        self.home = tmp_path / "home"
        (self.home / ".omnibase").mkdir(parents=True)
        self.env_file = self.home / ".omnibase" / "local.env"
        self.env_file.write_text(
            env_text
            if env_text is not None
            else f"REDPANDA_SUPERUSER_PASSWORD={_SUPERUSER_PASSWORD}\nONEX_TENANT_ID=stack-tenant\n"
        )

    def env(self, **extra: str) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("MAKE", "MFLAGS", "STUB_"))
        }
        env.pop("ONEX_TENANT_ID", None)
        env.update(
            PATH=f"{self.bin}:{os.environ.get('PATH', '')}",
            HOME=str(self.home),
            STUB_ROOT=str(self.root),
            STUB_SUPERUSER_PASSWORD=_SUPERUSER_PASSWORD,
        )
        env.update(extra)
        return env

    def command(self, env_file: Path | None = None) -> list[str]:
        return [
            "make",
            "--no-print-directory",
            "mint-local",
            f"LOCAL_ENV_FILE={env_file or self.env_file}",
        ]

    def run(
        self, env_file: Path | None = None, **extra: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.command(env_file),
            cwd=_REPO,
            env=self.env(**extra),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

    def mutations(self) -> list[str]:
        return (self.root / "mutations").read_text().splitlines()

    def calls(self) -> list[list[Any]]:
        return [
            json.loads(line)
            for line in (self.root / "calls.jsonl").read_text().splitlines()
        ]

    def stored(self) -> dict[str, str]:
        path = self.root / "store.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def logout(self) -> None:
        (self.root / "store.json").unlink()

    def lock(self, env_file: Path | None = None) -> Path:
        return (env_file or self.env_file).parent / ".mint-local.lock"


def _out(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


# --- AC2 / AC3 --------------------------------------------------------------------


def test_first_run_creates_stores_and_verifies(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    result = stack.run()
    assert result.returncode == 0, _out(result)
    principal = stack.stored()["username"]
    assert f"mint-local: principal {principal}" in result.stdout
    assert "mint-local: create" in result.stdout
    assert "mint-local: verified" in result.stdout
    assert len(stack.mutations()) == 1
    assert stack.stored()["lane"] == "local"


def test_second_run_is_unchanged_with_no_mutation(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    assert stack.run().returncode == 0
    again = stack.run()
    assert again.returncode == 0, _out(again)
    assert f"{stack.stored()['username']} unchanged" in again.stdout
    assert len(stack.mutations()) == 1, "the second run set a password"


def test_drift_is_refused_naming_the_extra_acl_with_the_password_untouched(
    tmp_path: Path,
) -> None:
    stack = _Stack(tmp_path)
    assert stack.run().returncode == 0
    principal = stack.stored()["username"]
    before = stack.stored()["password"]
    broker = json.loads((stack.root / "broker.json").read_text())
    broker["extra_acls"] = {principal: ["topic:stray(literal):read"]}
    (stack.root / "broker.json").write_text(json.dumps(broker))

    result = stack.run()
    assert result.returncode != 0
    assert "unchanged" not in result.stdout
    assert (
        "mint-local: ACL outside the plan: topic:stray(literal):read" in result.stderr
    )
    assert len(stack.mutations()) == 1
    assert stack.stored()["password"] == before
    assert (
        json.loads((stack.root / "broker.json").read_text())["users"][principal]
        == before
    )


def test_rotate_after_logout_sets_a_new_password_and_verifies(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    assert stack.run().returncode == 0
    first = stack.stored()["password"]
    stack.logout()
    result = stack.run()
    assert result.returncode == 0, _out(result)
    assert "mint-local: rotate" in result.stdout
    assert "mint-local: verified" in result.stdout
    assert len(stack.mutations()) == 2
    assert stack.stored()["password"] != first


def test_an_unsecured_broker_is_refused_before_any_node_call(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    result = stack.run(STUB_BROKER_OPEN="1")
    assert result.returncode != 0
    assert "run make up-local first" in result.stderr
    assert not any("onex" in call for call in stack.calls() if call[0] == "docker")
    assert stack.mutations() == []


# --- AC4: the lock ----------------------------------------------------------------


def test_lock_two_runs_started_together_make_one_mutation(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    env = stack.env(STUB_NODE_DELAY="1.5")
    runs = [
        subprocess.Popen(
            stack.command(),
            cwd=_REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    outcomes = [(run.wait(timeout=120), *run.communicate()) for run in runs]
    assert len(stack.mutations()) == 1, outcomes
    busy = [o for o in outcomes if "mint-local: busy, lock held by pid " in o[2]]
    assert len(busy) == 1, outcomes
    status = busy[0][0]
    assert status != 0
    winner = next(o for o in outcomes if o is not busy[0])
    assert winner[0] == 0, winner
    assert "mint-local: verified" in winner[1]
    assert not stack.lock().exists(), "the winner released the lock"


def test_lock_a_stale_lock_is_removed_and_said(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    dead = subprocess.Popen(["true"])
    dead.wait()
    lock = stack.lock()
    lock.mkdir()
    (lock / "pid").write_text(f"{dead.pid}\n")
    result = stack.run()
    assert result.returncode == 0, _out(result)
    assert f"removing a stale lock left by pid {dead.pid}" in result.stdout
    assert len(stack.mutations()) == 1
    assert not lock.exists()


def test_lock_a_live_lock_is_not_removed(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    holder = subprocess.Popen(["sleep", "60"])
    try:
        lock = stack.lock()
        lock.mkdir()
        (lock / "pid").write_text(f"{holder.pid}\n")
        result = stack.run()
        assert result.returncode != 0
        assert f"mint-local: busy, lock held by pid {holder.pid}" in result.stderr
        assert lock.is_dir(), "a busy run removed the live holder's lock"
        assert (lock / "pid").read_text().strip() == str(holder.pid)
        assert stack.mutations() == []
        assert not any(call[:2] == ["docker", "exec"] for call in stack.calls())
    finally:
        holder.kill()
        holder.wait()


def test_lock_a_fresh_lock_without_a_pid_is_busy(tmp_path: Path) -> None:
    """A holder between its mkdir and its pid write is live, not stale."""
    stack = _Stack(tmp_path)
    stack.lock().mkdir()
    result = stack.run()
    assert result.returncode != 0
    assert "mint-local: busy, lock held by pid unknown" in result.stderr
    assert stack.lock().is_dir()


def test_lock_is_released_after_a_failed_run(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    assert stack.run(STUB_NO_TERMINAL="1").returncode != 0
    assert not stack.lock().exists()


# --- AC5: the password ---------------------------------------------------------------


def test_no_secret_in_any_argument_output_or_input_file(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    first = stack.run()
    stack.logout()
    second = stack.run()
    assert first.returncode == 0 and second.returncode == 0
    secrets = {stack.stored()["password"]}
    secrets.update(
        json.loads((stack.root / "broker.json").read_text())["users"].values()
    )
    journal = (stack.root / "calls.jsonl").read_text()
    for secret in secrets:
        assert len(secret) == 64
        assert secret not in _out(first) + _out(second)
        assert secret not in journal, "a password was passed as an argument"
    # The input file and the node's state root are gone from the container.
    assert list((stack.root / "container").rglob("*.json")) == []


def test_no_secret_a_password_left_in_the_state_root_fails_the_run(
    tmp_path: Path,
) -> None:
    """Positive control for the state-root search: a leak is found and refused."""
    stack = _Stack(tmp_path)
    result = stack.run(STUB_LEAK_TO_STATE="1")
    assert result.returncode != 0
    assert "the password was found under the onex node state root" in result.stderr
    assert stack.stored() == {}, "a leaked password was still stored"


# --- AC7: the probe's tenant -----------------------------------------------------------


def test_tenant_the_probe_carries_the_stack_tenant_from_an_overridden_env_file(
    tmp_path: Path,
) -> None:
    stack = _Stack(tmp_path)
    stack.env_file.write_text(
        f"REDPANDA_SUPERUSER_PASSWORD={_SUPERUSER_PASSWORD}\nONEX_TENANT_ID=default-tenant\n"
    )
    other = tmp_path / "elsewhere" / "stack.env"
    other.parent.mkdir()
    other.write_text(
        f"REDPANDA_SUPERUSER_PASSWORD={_SUPERUSER_PASSWORD}\nONEX_TENANT_ID=stack-tenant\n"
    )
    result = stack.run(env_file=other, ONEX_TENANT_ID="native-tenant")
    assert result.returncode == 0, _out(result)
    probes = [call for call in stack.calls() if call[:2] == ["onex", "delegate"]]
    assert probes, "no host probe ran"
    assert {call[-1]["ONEX_TENANT_ID"] for call in probes} == {"stack-tenant"}
    assert stack.lock(other).parent == other.parent


def test_tenant_a_stack_without_one_does_not_invent_one(tmp_path: Path) -> None:
    stack = _Stack(
        tmp_path, env_text=f"REDPANDA_SUPERUSER_PASSWORD={_SUPERUSER_PASSWORD}\n"
    )
    result = stack.run()
    assert result.returncode == 0, _out(result)
    probes = [call for call in stack.calls() if call[:2] == ["onex", "delegate"]]
    assert {call[-1]["ONEX_TENANT_ID"] for call in probes} == {None}


# --- AC8: no terminal, no verified ------------------------------------------------------


def test_no_terminal_never_prints_verified(tmp_path: Path) -> None:
    stack = _Stack(tmp_path)
    result = stack.run(STUB_NO_TERMINAL="1")
    assert result.returncode != 0
    assert "verified" not in result.stdout
    assert "wrote no terminal event for correlation " in result.stderr


def test_no_terminal_a_valid_identity_without_a_terminal_is_not_unchanged(
    tmp_path: Path,
) -> None:
    stack = _Stack(tmp_path)
    assert stack.run().returncode == 0
    result = stack.run(STUB_NO_TERMINAL="1")
    assert "unchanged" not in result.stdout
    assert "verified" not in result.stdout
    assert result.returncode != 0


def test_wiring_the_target_is_phony_and_documented() -> None:
    makefile = (_REPO / "Makefile").read_text()
    phony = makefile.split(".PHONY:", 1)[1].split("\n\n", 1)[0]
    assert " mint-local " in phony.replace("\\\n", " ")
    assert "make mint-local" in makefile.split(".PHONY:", 1)[0]
