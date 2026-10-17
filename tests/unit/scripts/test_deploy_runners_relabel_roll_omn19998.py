# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Execution tests for ``deploy-runners.sh --relabel`` (OMN-19998).

The opt-in label ``omnibase-ci-multiarch`` reaches a LIVE runner only through a
re-registration, and the existing paths could not do it: ``--rolling`` judged
busy through the org registry (a cached read made all ten primary runners read
"busy or unknown", so a roll touched 0 of 10) and the secondary stand-up path
never touches an online runner. ``--relabel`` drains each runner on the
container's own evidence, re-registers it one at a time, and reads the pool back.

These tests run the real script end to end. ``ssh`` is a stub that runs the
remote command locally, ``rsync`` is a no-op, ``gh`` serves a registry kept in a
JSON state file, and ``docker`` is a stub over the same state file that models
containers, the compose render, stop/up and the registry effect of an ``up``.
Every assertion is on what the script DID (the stub's call log) rather than on
its source text.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"
FLEET_CONFIG = REPO_ROOT / "config" / "runner_fleet.yaml"

LABEL = "omnibase-ci-multiarch"
OLD_LABELS = "self-hosted,omnibase-ci,linux,x64,arch-amd64"
NEW_LABELS = f"self-hosted,omnibase-ci,{LABEL},linux,x64,arch-amd64"
SECRET = "AAAA-SUPPLIED-TOKEN-MUST-NEVER-SURFACE"
MINTED = "BBBB-MINTED-TOKEN-MUST-NEVER-SURFACE"

FAKE_DOCKER = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, sys

    state_path = os.environ["FAKE_STATE"]
    state = json.load(open(state_path))
    log = open(os.environ["FAKE_CALLS"], "a")
    args = sys.argv[1:]


    def save():
        json.dump(state, open(state_path, "w"))


    def custom(labels):
        return [
            {"name": x, "type": "custom"}
            for x in labels.split(",")
            if x not in ("self-hosted", "linux", "x64")
        ]


    if args[0] == "inspect":
        fmt, name = args[2], args[3]
        c = state["containers"][name]
        if ".State.Running" in fmt:
            print("true" if c["running"] else "false")
        else:
            print("health:" + c["health"])
    elif args[0] == "top":
        name = args[1]
        c = state["containers"][name]
        c["top_calls"] = c.get("top_calls", 0) + 1
        if c.get("busy_on_top_call") == c["top_calls"]:
            c["busy"] = True
        save()
        print("UID PID PPID CMD")
        print("1 1 0 Runner.Listener run")
        if c["busy"]:
            print("1 2 1 Runner.Worker spawnclient 1 2")
    elif args[:2] == ["image", "inspect"]:
        sys.exit(1 if state.get("image_missing") else 0)
    elif args[0] == "logs":
        c = state["containers"][args[-1]]
        print("2026-10-16 10:00:00Z: Job earlier completed with result: Succeeded")
        if c.get("log_busy") or c["busy"]:
            print("2026-10-16 10:05:00Z: Running job: pytest (shard 7/16)")
    elif args[0] == "stop":
        name = args[-1]
        log.write("STOP %s project=-\\n" % name)
        state["containers"][name]["running"] = False
        save()
    elif args[0] == "compose":
        verb_at = next(i for i, a in enumerate(args) if a in ("config", "stop", "up"))
        verb, rest = args[verb_at], args[verb_at + 1 :]
        project = args[args.index("-p") + 1] if "-p" in args else "-"
        name = rest[-1]
        c = state["containers"][name]
        if verb == "config":
            print("services:\\n  %s:\\n    image: omninode-runner:latest\\n    environment:\\n      RUNNER_LABELS: %s" % (name, c["rendered"]))
        elif verb == "stop":
            sys.exit("the relabel roll must stop with a bare docker stop")
        else:
            token = os.environ.get("RUNNER_TOKEN") or os.environ.get("FAKE_OTHER_TOKEN_VAR", "")
            log.write(
                "UP %s project=%s flags=%s token_present=%s token_in_argv=%s\\n"
                % (name, project, " ".join(rest[:-1]), bool(token), any(token and token in a for a in args))
            )
            log.write("TOKEN_SEEN %s\\n" % token)
            c["running"] = True
            c["health"] = "starting" if c.get("stuck") else "healthy"
            for r in state["registry"]:
                if r["name"] == name:
                    r["status"] = "online"
                    r["labels"] = [{"name": n, "type": "custom"} for n in
                                   [x["name"] for x in custom(c["rendered"])]]
            save()
    else:
        sys.exit("fake docker: unsupported " + repr(args))
    """
)

FAKE_GH = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, sys

    args = sys.argv[1:]
    log = open(os.environ["FAKE_CALLS"], "a")
    if "POST" in args:
        log.write("GH_POST\\n")
        print(os.environ["FAKE_MINTED_TOKEN"])
        sys.exit(0)
    log.write("GH_READ %s\\n" % " ".join(a for a in args if a.startswith("/")))
    state = json.load(open(os.environ["FAKE_STATE"]))
    print(json.dumps({"total_count": len(state["registry"]), "runners": state["registry"]}))
    """
)


@dataclass
class World:
    root: Path
    state: Path
    calls: Path

    def run(
        self, *flags: str, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        merged = dict(os.environ)
        merged.update(
            {
                "PATH": f"{self.root / 'bin'}:{os.environ['PATH']}",
                "FAKE_STATE": str(self.state),
                "FAKE_CALLS": str(self.calls),
                "FAKE_MINTED_TOKEN": MINTED,
                "ROLL_ONLINE_INTERVAL_SECONDS": "1",
                "ROLL_ONLINE_MAX_SECONDS": "3",
                "RELABEL_DRAIN_INTERVAL_SECONDS": "1",
                "RELABEL_READBACK_INTERVAL_SECONDS": "0",
                "HOME": str(self.root / "home"),
            }
        )
        merged.pop("DEPLOY_RUNNER_TOKEN", None)
        merged.update(env or {})
        return subprocess.run(
            ["bash", str(DEPLOY_SCRIPT), *flags],
            env=merged,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

    def calls_text(self) -> str:
        return self.calls.read_text(encoding="utf-8") if self.calls.exists() else ""

    def lines(self, prefix: str) -> list[str]:
        return [x for x in self.calls_text().splitlines() if x.startswith(prefix)]

    def registry(self) -> dict[str, dict]:
        data = json.loads(self.state.read_text(encoding="utf-8"))
        return {r["name"]: r for r in data["registry"]}

    def set_state(self, **fields: object) -> None:
        data = json.loads(self.state.read_text(encoding="utf-8"))
        data.update(fields)
        self.state.write_text(json.dumps(data), encoding="utf-8")

    def mutate(self, name: str, **fields: object) -> None:
        data = json.loads(self.state.read_text(encoding="utf-8"))
        data["containers"][name].update(fields)
        self.state.write_text(json.dumps(data), encoding="utf-8")


def _registry_row(name: str, labels: str) -> dict:
    custom = [
        {"name": x, "type": "custom"}
        for x in labels.split(",")
        if x not in ("self-hosted", "linux", "x64")
    ]
    return {
        "name": name,
        "status": "online",
        "busy": False,
        "labels": [{"name": "self-hosted", "type": "read-only"}, *custom],
    }


def _build_world(tmp_path: Path, prefix: str, count: int) -> World:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (tmp_path / "home").mkdir()
    ssh = bindir / "ssh"
    # A remote mkdir is recorded, not run: the host path (/home/jonah/...) does
    # not exist, and may not be created, on a hosted CI runner. docker and gh
    # are shell functions naming the stubs, because the secondary-host PATH
    # preamble puts /usr/local/bin and /opt/homebrew/bin ahead of the stub dir.
    ssh.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nshift\n"
        f'docker() {{ "{bindir}/docker" "$@"; }}; gh() {{ "{bindir}/gh" "$@"; }}\n'
        "export -f docker gh\n"
        'exec bash -c \'mkdir() { echo "MKDIR $*" >> "$FAKE_CALLS"; }; \'"$1"\n'
    )
    rsync = bindir / "rsync"
    rsync.write_text('#!/usr/bin/env bash\necho "RSYNC $*" >> "$FAKE_CALLS"\nexit 0\n')
    (bindir / "docker").write_text(FAKE_DOCKER)
    (bindir / "gh").write_text(FAKE_GH)
    for f in bindir.iterdir():
        f.chmod(f.stat().st_mode | stat.S_IEXEC)

    names = [f"{prefix}-{i}" for i in range(1, count + 1)]
    state = {
        "containers": {
            n: {
                "running": True,
                "busy": False,
                "health": "healthy",
                "rendered": NEW_LABELS,
            }
            for n in names
        },
        # An unrelated runner and a surplus pool member: the pool count must
        # include the second and ignore the first.
        "registry": [_registry_row(n, OLD_LABELS) for n in names]
        + [
            _registry_row(f"{prefix}-18", OLD_LABELS),
            _registry_row("some-other-runner-1", OLD_LABELS),
        ],
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return World(root=tmp_path, state=state_path, calls=tmp_path / "calls.log")


@pytest.fixture
def primary(tmp_path: Path) -> World:
    return _build_world(tmp_path, "omninode-runner", 10)


def test_the_script_parses_and_documents_the_mode() -> None:
    parse = subprocess.run(
        ["bash", "-n", str(DEPLOY_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert parse.returncode == 0, parse.stderr
    header = DEPLOY_SCRIPT.read_text(encoding="utf-8").split("set -euo pipefail")[0]
    assert "--relabel" in header and "--drain-timeout" in header


@pytest.mark.parametrize("other", ["--rolling", "--soft", "--retire-surplus"])
def test_relabel_refuses_to_combine_with_the_other_modes(
    primary: World, other: str
) -> None:
    result = primary.run("--relabel", other, "--dry-run")
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr
    assert primary.lines("STOP") == [] and primary.lines("UP") == []


def test_dry_run_changes_nothing_and_names_the_idle_and_busy_runners(
    primary: World,
) -> None:
    primary.mutate("omninode-runner-2", busy=True)
    result = primary.run("--relabel", "--dry-run")
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert primary.lines("STOP") == [] and primary.lines("UP") == []
    assert primary.lines("GH_POST") == [], "a dry run must not mint a token"
    assert len(primary.lines("GH_READ")) == 1, "one registry read, no per-runner polls"
    assert "[DRY RUN] omninode-runner-1 (online) is idle" in result.stdout
    assert "omninode-runner-2 (online) is idle" not in result.stdout
    assert "9 runner(s) idle and relabellable now" in result.stdout
    assert "1 busy" in result.stdout
    # the registry itself is untouched
    assert all(LABEL not in str(r["labels"]) for r in primary.registry().values())


def test_a_real_roll_relabels_the_idle_runners_one_at_a_time_and_keeps_the_count(
    primary: World,
) -> None:
    primary.mutate("omninode-runner-2", busy=True)
    result = primary.run("--relabel", "--drain-timeout=1")
    assert result.returncode == 0, (result.stdout, result.stderr)

    ups = primary.lines("UP")
    assert [u.split()[1] for u in ups] == [
        f"omninode-runner-{i}" for i in (1, 3, 4, 5, 6, 7, 8, 9, 10)
    ], "the busy runner must be left alone and the rest taken in order"
    for line in ups:
        assert "--force-recreate" in line and "--no-deps" in line
        assert "--pull never" in line
        assert "--remove-orphans" not in line
        assert "token_present=True" in line and "token_in_argv=False" in line
    # strictly serial: every stop is immediately followed by its own up
    order = [
        " ".join(x.split()[:2])
        for x in primary.calls_text().splitlines()
        if x.startswith(("STOP", "UP"))
    ]
    for i in range(0, len(order), 2):
        stop, up = order[i], order[i + 1]
        assert stop.startswith("STOP ") and up.startswith("UP ")
        assert stop.split()[1] == up.split()[1]

    assert len(primary.lines("GH_POST")) == 9, "one fresh token per relabelled runner"
    assert len(primary.lines("GH_READ")) <= 10, "registry reads stay under the budget"
    registry = primary.registry()
    assert LABEL not in str(registry["omninode-runner-2"]["labels"])
    assert LABEL in str(registry["omninode-runner-1"]["labels"])
    assert len(registry) == 12, "no runner registered or removed"
    assert "still busy: omninode-runner-2" in result.stderr
    assert "count_before=11 count_after=11" in result.stdout


def test_only_the_compose_declarations_are_synced(primary: World) -> None:
    """The containers bind-mount the entrypoint and healthcheck; a relabel must
    not replace those on a host whose image is older than the repo's copy.
    """
    result = primary.run("--relabel", "--only=omninode-runner-3")
    assert result.returncode == 0, (result.stdout, result.stderr)
    synced = " ".join(primary.lines("RSYNC"))
    assert "docker-compose.runners.yml" in synced
    for artifact in (
        "entrypoint.sh",
        "healthcheck.sh",
        "runner-monitor.sh",
        "Dockerfile",
    ):
        assert artifact not in synced


def test_a_busy_runner_is_never_stopped(primary: World) -> None:
    for i in range(1, 11):
        primary.mutate(f"omninode-runner-{i}", busy=True)
    result = primary.run("--relabel", "--drain-timeout=1")
    assert primary.lines("STOP") == [] and primary.lines("UP") == []
    assert primary.lines("GH_POST") == [], "no token is minted for a runner left alone"
    assert "still busy" in result.stderr


def test_a_token_never_reaches_output_or_argv(primary: World) -> None:
    result = primary.run(
        "--relabel", "--only=omninode-runner-3", env={"DEPLOY_RUNNER_TOKEN": SECRET}
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert primary.lines("GH_POST") == [], "a supplied token is used instead of minting"
    seen = primary.lines("TOKEN_SEEN")
    assert seen == [f"TOKEN_SEEN {SECRET}"], "the runner's recreate received the token"
    assert "token_in_argv=False" in primary.lines("UP")[0]
    for text in (result.stdout, result.stderr):
        assert SECRET not in text and MINTED not in text


def test_a_minted_token_never_reaches_output(primary: World) -> None:
    result = primary.run("--relabel", "--only=omninode-runner-3")
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert primary.lines("GH_POST") == ["GH_POST"]
    assert primary.lines("TOKEN_SEEN") == [f"TOKEN_SEEN {MINTED}"]
    assert MINTED not in result.stdout and MINTED not in result.stderr


def test_a_runner_whose_listener_log_shows_a_job_in_hand_is_left_alone(
    primary: World,
) -> None:
    """The moment between a job being handed over and its Runner.Worker process
    appearing reads idle on the process list; the listener's own log does not.
    """
    primary.mutate("omninode-runner-3", log_busy=True)
    result = primary.run("--relabel", "--drain-timeout=1")
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "omninode-runner-3" not in " ".join(primary.lines("STOP"))
    assert "omninode-runner-3" not in " ".join(primary.lines("UP"))
    assert "still busy: omninode-runner-3" in result.stderr


def test_a_missing_image_tag_refuses_the_roll_before_any_runner_is_stopped(
    primary: World,
) -> None:
    """A recreate that cannot find its image would stop the runner and then fail
    to start it; the roll must notice before the first stop.
    """
    primary.set_state(image_missing=True)
    result = primary.run("--relabel", "--drain-timeout=1")
    assert result.returncode != 0
    assert "is not present on" in result.stderr
    assert primary.lines("STOP") == [] and primary.lines("UP") == []
    assert primary.lines("GH_POST") == []


def test_a_job_that_lands_between_the_idle_read_and_the_stop_wins(
    primary: World,
) -> None:
    """The idle check is repeated on the host immediately before the stop. A
    runner that takes a job in that window must be left running.
    """
    primary.mutate("omninode-runner-3", busy_on_top_call=2)
    result = primary.run("--relabel", "--drain-timeout=1")
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "omninode-runner-3" not in " ".join(primary.lines("STOP"))
    assert "omninode-runner-3" not in " ".join(primary.lines("UP"))
    assert "still busy: omninode-runner-3" in result.stderr
    assert len(primary.lines("UP")) == 9


def test_a_runner_that_does_not_return_halts_the_roll(primary: World) -> None:
    primary.mutate("omninode-runner-3", stuck=True)
    result = primary.run("--relabel", "--drain-timeout=1")
    assert result.returncode != 0
    assert "HALTED at omninode-runner-3" in result.stderr
    assert "do NOT retry blindly" in result.stderr
    touched = [u.split()[1] for u in primary.lines("UP")]
    assert touched == ["omninode-runner-1", "omninode-runner-2", "omninode-runner-3"], (
        "the roll must stop at the runner that did not return"
    )


def test_a_runner_whose_compose_lacks_the_label_is_left_untouched(
    primary: World,
) -> None:
    primary.mutate("omninode-runner-1", rendered="self-hosted,omnibase-ci,linux,x64")
    result = primary.run("--relabel", "--drain-timeout=1")
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "omninode-runner-1" not in [u.split()[1] for u in primary.lines("UP")]
    assert "lacks omnibase-ci-multiarch" in result.stderr
    assert "refused=1" in result.stdout


def test_a_relabel_may_not_drop_a_label_the_runner_holds_today(
    primary: World,
) -> None:
    primary.mutate("omninode-runner-4", rendered=f"self-hosted,{LABEL},linux,x64")
    result = primary.run("--relabel", "--drain-timeout=1")
    assert "omninode-runner-4" not in [u.split()[1] for u in primary.lines("UP")]
    assert "would drop" in result.stderr


def test_a_runner_already_online_with_the_label_is_not_touched(
    primary: World,
) -> None:
    data = json.loads(primary.state.read_text(encoding="utf-8"))
    for r in data["registry"]:
        if r["name"] == "omninode-runner-5":
            r["labels"].append({"name": LABEL, "type": "custom"})
    primary.state.write_text(json.dumps(data), encoding="utf-8")
    result = primary.run("--relabel", "--drain-timeout=1")
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "omninode-runner-5" not in [u.split()[1] for u in primary.lines("UP")]
    assert "already=1" in result.stdout


def test_limit_counts_relabelled_runners_and_only_names_one(primary: World) -> None:
    result = primary.run("--relabel", "--limit=2")
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert [u.split()[1] for u in primary.lines("UP")] == [
        "omninode-runner-1",
        "omninode-runner-2",
    ]
    bad = primary.run("--relabel", "--only=omninode-runner-99", "--dry-run")
    assert bad.returncode != 0 and "is not a runner of" in bad.stderr


def test_registry_reads_are_capped_by_the_budget(primary: World) -> None:
    result = primary.run("--relabel", "--dry-run", env={"RELABEL_API_READ_BUDGET": "0"})
    assert result.returncode != 0
    assert "budget" in result.stderr
    assert primary.lines("STOP") == [] and primary.lines("UP") == []
    assert primary.lines("GH_READ") == []


def _declared_secondary_action_pool() -> tuple[str, str, int]:
    """The (host, pool prefix, runner count) of a declared non-primary amd64
    host whose pool carries the multiarch class, read from the fleet config so
    the test follows the inventory instead of spelling a host.
    """
    config = yaml.safe_load(FLEET_CONFIG.read_text(encoding="utf-8"))
    for row in config["hosts"]:
        if row["host"] == config["runner_host"]:
            continue
        for pool in row.get("pools", []):
            if "action-multiarch" in pool["classes"] and row["arch"] == "amd64":
                return row["host"], pool["runner_name_prefix"], pool["expected_count"]
    raise AssertionError("no secondary amd64 pool declares action-multiarch")


def test_a_secondary_pool_is_rolled_under_its_own_compose_project(
    tmp_path: Path,
) -> None:
    host, prefix, count = _declared_secondary_action_pool()
    world = _build_world(tmp_path, prefix, count)
    world.mutate(f"{prefix}-7", busy=True)
    result = world.run(
        "--relabel", f"--host={host}", f"--pool={prefix}", "--drain-timeout=1"
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    ups = world.lines("UP")
    assert len(ups) == count - 1 and all(f"project={prefix}" in u for u in ups)
    assert f"{prefix}-7" not in " ".join(ups)
    assert f"count_before={count + 1} count_after={count + 1}" in result.stdout
