# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15509 -- the .201 system-health Slack reporter must see every lane's runtime.

WHAT IS UNDER TEST
    ``deploy/maintenance/omninode-system-slack-report.sh`` -- the real bash
    artifact that root runs on ``.201`` via
    ``/etc/cron.d/omninode-system-slack-report``. Every test here drives that
    file itself, not a Python re-implementation of it: a surrogate would prove
    nothing about the thing that actually alarms (memory
    ``feedback_test_the_artifact_that_runs``).

THE RED-BEFORE IS REAL, NOT ASSERTED
    ``tests/fixtures/omn15509/omninode-system-slack-report.as-deployed-20260730.sh``
    is a byte-for-byte capture of the version that was live on ``.201`` during
    the 2026-07-30T16:19-16:45Z outage. ``test_as_deployed_reports_green_on_the
    _replayed_outage`` drives THAT file against the replayed outage state and
    asserts it reports the dev runtime nowhere and every runtime endpoint as
    HTTP 200 -- the false green. The paired test drives the fixed file against
    the identical state and asserts CRITICAL naming the dev runtime.

HERMETICITY
    ``docker``/``curl``/``df``/``hostname``/``sha256sum``/``flock`` are replaced
    by stubs on PATH. Both scripts pin ``PATH=`` at the top, so the harness
    injects one identical line after that assignment in BOTH scripts. The
    transformation is symmetric by construction (same helper, same regex), so a
    difference in outcome can only come from the scripts' own logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXED_SCRIPT = REPO_ROOT / "deploy" / "maintenance" / "omninode-system-slack-report.sh"
AS_DEPLOYED_SCRIPT = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn15509"
    / "omninode-system-slack-report.as-deployed-20260730.sh.captured"
)
# sha256 of the copy read off .201:/data/maintenance/bin/omninode-system-slack-report.sh
# at 2026-07-30T17:0xZ. Asserted below: if anyone edits the fixture, the
# "byte-for-byte capture" claim stops being true and the RED-before proof
# stops meaning anything, so the edit must fail loudly rather than pass.
# The `.captured` suffix keeps the SPDX-header hook off the file for the same
# reason -- a stamped header would no longer be the artifact that ran.
AS_DEPLOYED_SHA256 = "5fe6e5a61d6074922142006f5fc905e146bc4a1dcc18dbe4e0da99ddaec209da"
CRON_UNIT = (
    REPO_ROOT / "deploy" / "maintenance" / "cron.d" / "omninode-system-slack-report"
)
RUNTIME_POLICY_ENV = REPO_ROOT / "docker" / "runtime-policy.env"

# Lane -> the runtime-policy.env key that carries its MAIN runtime port.
#
# DERIVED from the rendered policy, never hand-written (OMN-15556). The previous
# revision carried a literal dev/stability-test/prod dict -- the *same* three
# rows the reporter enumerated -- while claiming in this very comment that the
# map "is asserted against the rendered policy rather than trusted". Nothing
# read the policy, so the guard was structurally blind to the one regression it
# exists to catch: a lane declared in runtime-policy.env that nothing probes.
# JUDGE_RUNTIME_MAIN_PORT (:48085, seven containers live on .201) sat unprobed
# behind a fully green suite. Deriving the map means the next lane added to the
# policy fails this module until the reporter enumerates it.
LANE_MAIN_PORT_KEY_RE = re.compile(r"^([A-Z0-9_]+)_RUNTIME_MAIN_PORT=")


def _derive_lane_port_keys(policy: Path) -> dict[str, str]:
    """Parse {lane: policy_key} out of the rendered runtime policy.

    Lane name is the key prefix lowercased with underscores hyphenated, which is
    the label convention the reporter uses in RUNTIME_LANE_SPECS
    (STABILITY_TEST_RUNTIME_MAIN_PORT -> stability-test).
    """
    keys: dict[str, str] = {}
    for line in policy.read_text().splitlines():
        match = LANE_MAIN_PORT_KEY_RE.match(line.strip())
        if match is None:
            continue
        prefix = match.group(1)
        keys[prefix.lower().replace("_", "-")] = f"{prefix}_RUNTIME_MAIN_PORT"
    # Fail loudly rather than deriving an empty map: an empty map would make
    # every lane assertion below vacuously true, which is the failure mode this
    # whole module exists to prevent.
    assert keys, f"no *_RUNTIME_MAIN_PORT keys found in {policy}"
    return keys


LANE_PORT_KEYS = _derive_lane_port_keys(RUNTIME_POLICY_ENV)

# The reporter declares the same mapping in bash; parsed here so the two can be
# held in two-way parity by a test rather than by convention.
RUNTIME_LANE_SPECS_RE = re.compile(
    r"^RUNTIME_LANE_SPECS=\((?P<body>.*?)^\)", re.MULTILINE | re.DOTALL
)
# OMN-17150: the sibling table of lanes the reporter deliberately does NOT probe.
# Its whole purpose is to keep "not probed" a declaration instead of an absence:
# the parity assertion below is over the UNION of the two tables, so a lane can
# never leave this file silently, only move visibly between the two.
RUNTIME_LANE_UNPROBED_RE = re.compile(
    r"^RUNTIME_LANE_UNPROBED=\((?P<body>.*?)^\)", re.MULTILINE | re.DOTALL
)

#: Lanes that MUST stay in the probed table. Every lane the platform itself is
#: responsible for keeping up. Named explicitly rather than derived, because the
#: whole risk the unprobed table introduces is that one of these gets quietly
#: moved into it -- which is the OMN-15556 judge blind spot with an extra step.
#:
#: "prod" was removed from this set 2026-09-16 (OMN-18320). It is not the
#: OMN-15556 blind-spot risk this set exists to catch: the compose project it
#: named (omnibase-infra-prod) was shut down 2026-09-13 and no longer exists on
#: .201, so PROD_RUNTIME_MAIN_PORT resolving to a dead endpoint is not a lane
#: the platform is "responsible for keeping up" -- there is nothing to keep up.
#: AWS onex-prod is the real production runtime and this reporter has never
#: probed it. docker/runtime-policy.env still renders the key (OMN-18320 left
#: it and docker-compose.prod.yml in place deliberately), which is exactly why
#: it still has to be declared -- in RUNTIME_LANE_UNPROBED, not here.
MUST_BE_PROBED_LANES = frozenset({"dev", "stability-test", "judge"})


def _parse_lane_table(script: Path, pattern: re.Pattern[str]) -> dict[str, str]:
    """Parse one of the reporter's ``lane|policy-key`` bash arrays."""
    match = pattern.search(script.read_text())
    if match is None:
        return {}
    specs: dict[str, str] = {}
    for row in re.findall(r'"([^"]+)"', match.group("body")):
        lane, _, key = row.partition("|")
        specs[lane] = key
    return specs


def _script_lane_specs(script: Path) -> dict[str, str]:
    """Parse the reporter's own RUNTIME_LANE_SPECS array into {lane: policy_key}."""
    specs = _parse_lane_table(script, RUNTIME_LANE_SPECS_RE)
    assert specs, f"RUNTIME_LANE_SPECS in {script} is missing or parsed to an empty map"
    return specs


def _script_unprobed_lane_specs(script: Path) -> dict[str, str]:
    """Parse RUNTIME_LANE_UNPROBED. Absent (or empty) is legal -- it means every
    policy lane is probed, which is the stronger state."""
    return _parse_lane_table(script, RUNTIME_LANE_UNPROBED_RE)


#: The lanes this reporter is expected to actually probe: every policy lane the
#: script has not explicitly declared unprobed. Derived, so it tracks the script.
UNPROBED_LANE_PORT_KEYS = _script_unprobed_lane_specs(FIXED_SCRIPT)
PROBED_LANE_PORT_KEYS = {
    lane: key
    for lane, key in LANE_PORT_KEYS.items()
    if lane not in UNPROBED_LANE_PORT_KEYS
}


pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("jq") is None,
    reason="bash + jq are required to drive the real reporter artifact",
)


def test_as_deployed_fixture_is_the_unmodified_201_capture() -> None:
    """Provenance guard for the RED-before artifact."""
    digest = hashlib.sha256(AS_DEPLOYED_SCRIPT.read_bytes()).hexdigest()
    assert digest == AS_DEPLOYED_SHA256, (
        "the as-deployed fixture no longer matches the copy captured from "
        ".201:/data/maintenance/bin/omninode-system-slack-report.sh on 2026-07-30; "
        "the RED-before proof is only meaningful against the unmodified artifact"
    )


def _policy_port(key: str) -> str:
    """Read one key out of the rendered runtime policy, same idiom as the script."""
    value = ""
    for line in RUNTIME_POLICY_ENV.read_text().splitlines():
        if line.startswith(f"{key}="):
            value = line.split("=", 1)[1].strip().strip("\"'")
    assert value, f"{key} missing from {RUNTIME_POLICY_ENV}"
    return value


def _write(path: Path, body: str, *, executable: bool = False) -> None:
    path.write_text(body)
    if executable:
        path.chmod(0o755)


def _make_stub_bin(
    tmp_path: Path,
    *,
    http: dict[str, tuple[int, str]],
    docker_state: dict[str, Any],
) -> Path:
    """Build a stub bin dir. ``http`` maps port -> (status_code, body)."""
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir()
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {"http": {str(k): v for k, v in http.items()}, "docker": docker_state}
        )
    )

    # curl: honours -o <file> and -w '%{http_code}'. A port absent from the spec
    # is a connection failure (empty output, non-zero exit) exactly like a real
    # refused connection, so "endpoint not reachable" is never silently a pass.
    _write(
        bin_dir / "curl",
        f"""#!/usr/bin/env bash
SPEC={spec}
out=""; url=""
args=("$@")
for ((i=0; i<${{#args[@]}}; i++)); do
  case "${{args[$i]}}" in
    -o) out="${{args[$((i+1))]}}" ;;
    http://*|https://*) url="${{args[$i]}}" ;;
  esac
done
port=$(sed -E 's|.*:([0-9]+)/.*|\\1|' <<<"$url")
read -r code body < <(python3 - "$SPEC" "$port" <<'PY'
import json,sys
spec=json.load(open(sys.argv[1]))
e=spec["http"].get(sys.argv[2])
print("NONE","" ) if e is None else print(e[0], json.dumps(e[1]))
PY
)
if [[ "$code" == "NONE" ]]; then
  [[ -n "$out" ]] && : >"$out"
  printf '%s' ''
  exit 7
fi
decoded=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]))' "$body")
[[ -n "$out" ]] && printf '%s' "$decoded" >"$out"
printf '%s' "$code"
exit 0
""",
        executable=True,
    )

    _write(
        bin_dir / "docker",
        f"""#!/usr/bin/env bash
SPEC={spec}
python3 - "$SPEC" "$@" <<'PY'
import json,sys
spec=json.load(open(sys.argv[1]))["docker"]
a=sys.argv[2:]
argline=" ".join(a)
def ps_all():
    return [(c["name"], c["status"]) for c in spec["containers"]]
def running():
    return [c for c in spec["containers"] if c["status"].startswith("Up")]
if a[0]=="ps" and "status=created" in argline:
    for c in spec["containers"]:
        if c["status"].lower().startswith("created"): print(c["name"])
elif a[0]=="ps" and "-a" in a:
    for n,s in ps_all(): print(f"{{n}}\\t{{s}}")
elif a[0]=="ps" and ".Ports" in " ".join(a):
    for c in running(): print(c["name"] + "\\t" + c.get("ports",""))
elif a[0]=="ps" and "health=starting" in " ".join(a):
    for c in running():
        if "health: starting" in c["status"]: print(c["name"])
elif a[0]=="ps":
    for c in running(): print(c["name"])
elif a[0]=="inspect":
    name=a[-1]
    c=next((c for c in spec["containers"] if c["name"]==name), None)
    if c is None: sys.exit(1)
    fmt=a[a.index("-f")+1]
    if ".Created" in fmt:
        L=c.get("labels") or {{}}
        print("|".join([c.get("created_at",""),
                        L.get("com.docker.compose.project",""),
                        L.get("onex.cleanup-owner",""),
                        L.get("onex.purpose","")]))
    elif "StartedAt" in fmt: print(c["started_at"])
    else: print(c.get("start_period_ns",0))
elif a[0]=="volume":
    for v in spec.get("dangling", []): print(v)
PY
""",
        executable=True,
    )

    _write(
        bin_dir / "df",
        """#!/usr/bin/env bash
echo "Filesystem 1G-blocks Used Avail Use% Mounted"
echo "target 1832G 75G 1664G 5%"
""",
        executable=True,
    )
    _write(
        bin_dir / "hostname", "#!/usr/bin/env bash\necho omninode-pc\n", executable=True
    )
    _write(bin_dir / "flock", "#!/usr/bin/env bash\nexit 0\n", executable=True)
    if shutil.which("sha256sum") is None:
        _write(
            bin_dir / "sha256sum",
            "#!/usr/bin/env bash\nshasum -a 256\n",
            executable=True,
        )
    return bin_dir


def _stage(script: Path, tmp_path: Path, bin_dir: Path) -> Path:
    """Copy ``script`` and redirect it at the sandbox.

    Two textual transformations, applied by the SAME code to the fixed script
    and to the as-deployed fixture so neither is advantaged:

    1. inject the stub bin after the pinned ``PATH=`` assignment;
    2. repoint the ``/data/maintenance`` state/log/lock paths at ``tmp_path``.

    (2) is only needed because the as-deployed version hardcodes those paths
    with no env override -- the fixed version reads them from the environment,
    so for it the rewrite is equivalent to the env vars ``_run`` already sets.
    Neither transformation touches probe selection, status classification, or
    message formatting, which is all these tests assert on.
    """
    staged = tmp_path / f"staged-{script.name}"
    lines = script.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("PATH="):
            lines.insert(index + 1, f'PATH="{bin_dir}:$PATH"\n')
            break
    else:  # pragma: no cover - both artifacts pin PATH; a miss is a real defect
        raise AssertionError(f"no PATH assignment found in {script}")
    patched = "".join(lines)
    sandbox = tmp_path / "sandbox"
    for var, sub in (
        ("STATE_DIR", sandbox / "state"),
        ("LOG_DIR", sandbox / "logs"),
        ("LOCK_FILE", sandbox / "lock"),
        ("ENV_FILE", sandbox / "absent.env"),
    ):
        patched = re.sub(
            rf"^{var}=.*$", f"{var}={sub}", patched, count=1, flags=re.MULTILINE
        )
    staged.write_text(patched)
    staged.chmod(0o755)
    return staged


def _run(
    script: Path,
    tmp_path: Path,
    bin_dir: Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> str:
    env = dict(os.environ)
    env.update(
        {
            "OMNINODE_ALERT_ENV_FILE": str(tmp_path / "absent.env"),
            "OMNINODE_ALERT_STATE_DIR": str(tmp_path / "state"),
            "OMNINODE_ALERT_LOG_DIR": str(tmp_path / "logs"),
            "OMNINODE_ALERT_LOCK_FILE": str(tmp_path / "lock"),
            "OMNINODE_INFRA_REPO_ROOT": str(REPO_ROOT),
            "OMNINODE_RUNTIME_POLICY_ENV": str(RUNTIME_POLICY_ENV),
            "SLACK_BOT_TOKEN": "test-token",
            "SLACK_CHANNEL_ID": "C-TEST",
            # OMN-15550: the reporter now shells out to the required-context
            # probe from collect(). These lane tests are about disk/docker/
            # endpoint classification and must not acquire a GitHub network
            # dependency; the probe's own rows are asserted in
            # test_omninode_ci_required_context_probe.py.
            "OMNINODE_CI_PROBE_ENABLED": "0",
            # OMN-18567: same reasoning one row up. The reporter also reads
            # the runner-tree converge verdict from collect(). These tests
            # are about disk/docker/endpoint classification and must not
            # acquire a dependency on a state file that a cron tick writes;
            # the collector's own rows are asserted in the OMN-18567 block
            # at the end of this file, which switches it back on.
            "OMNINODE_RUNNER_TREE_CHECK_ENABLED": "0",
            # OMN-18944: same reasoning again. The reporter also reads the
            # backup freshness gate's latest run from collect(). These tests
            # must not acquire a GitHub network dependency; that check's own
            # rows are asserted in
            # tests/scripts/test_postgres_backup_freshness_row_omn18944.py,
            # which drives it through its fetch seam against recorded payloads.
            "OMNINODE_BACKUP_GATE_CHECK_ENABLED": "0",
        }
    )
    if extra_env:
        env.update(extra_env)
    staged = _stage(script, tmp_path, bin_dir)
    proc = subprocess.run(
        ["bash", str(staged), "--mode", "dry-run"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    out = proc.stdout
    if not out.strip():
        # The as-deployed script redirects stdout into its log file even in
        # dry-run; read the report back from there.
        logs = sorted((tmp_path / "sandbox" / "logs").glob("*.log"))
        if logs:
            out = logs[-1].read_text()
    assert out.strip(), (
        f"no report produced: rc={proc.returncode} stderr={proc.stderr[-2000:]}"
    )
    return out


# --------------------------------------------------------------------------
# The 2026-07-30T16:19-16:45Z outage state, replayed verbatim.
#   dev :8085      -> 503, healthy=false, is_running=false, no handlers
#   dev :8086      -> connection refused (container never started)
#   stability 18085-> 200 healthy
#   prod 28085     -> 200 healthy
#   everything else 200, all infra containers healthy
# --------------------------------------------------------------------------
DEV_503_BODY = json.dumps(
    {
        "status": "unhealthy",
        "healthy": False,
        "is_running": False,
        "registered_handlers": [],
        "config_prefetch_status": "pending",
    }
)
# The healthy runtime body -- A REAL CAPTURE, not a reconstruction (OMN-15547).
#
# History of this one fixture is the whole argument for the incident-replay
# convention:
#
#   * ORIGINALLY it was `{"status":"healthy","healthy":true,"version":"0.38.4"}`
#     -- 63 bytes. `check_runtime_lane` truncated the body to 180 bytes before
#     handing it to jq, so under a 63-byte fixture the truncation never bit. The
#     654-line suite was green while the deployed artifact reported CRITICAL for
#     all three lanes against a fully healthy fleet (OMN-15525).
#   * THE OMN-15525 FIX replaced it with a larger literal whose own comment read
#     "Shape mirrors the live body". That crosses the byte boundary, but it is
#     still something a person typed: it can only exhibit the failure modes its
#     author already thought of. Key names, nesting depth and the actual value
#     shapes were guesses.
#   * NOW it is the bytes off the wire. Anything the real payload does that a
#     reconstruction would not -- ordering, unicode, numeric formatting, a key
#     nobody remembered -- is in scope for this suite by construction.
#
# Registered as an incident replay case in tests/incident_replays/registry.yaml
# (`omn15525-health-body-truncation`, regression_class: false_red).
HEALTHY_BODY_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "omn15547" / "health-dev-8085.json.captured"
)
# sha256 of the response body returned by GET http://127.0.0.1:8085/health on
# .201 at 2026-07-31T00:03:06Z (dev lane, live-healthy at the time). Asserted
# below for the same reason the as-deployed script capture is: an edited
# artifact is no longer the artifact, and the claim must break loudly.
HEALTHY_BODY_SHA256 = "240178e33079f76b38f4995c39b7a90da68c37d4f08365c13752411a9da6050a"
HEALTHY_BODY = HEALTHY_BODY_FIXTURE.read_text(encoding="utf-8")
# Guard the guard: if someone shrinks this fixture the truncation defect becomes
# invisible again, so assert the property the fixture exists to provide.
assert len(HEALTHY_BODY) > 180, (
    "HEALTHY_BODY must exceed the reporter's display-excerpt limit or the "
    "OMN-15525 truncation regression cannot be observed"
)


def _outage_http(lane_ports: dict[str, str]) -> dict[str, tuple[int, str]]:
    """The replayed outage: dev 503, every OTHER declared lane healthy.

    Built from the derived lane set rather than a fixed dev/stability/prod
    triple (OMN-15556). A lane absent from this stub reads as connection-refused
    and would fabricate an outage the fixture is not replaying, so a
    newly-declared lane has to land here as green automatically.
    """
    http: dict[str, tuple[int, str]] = dict.fromkeys(
        lane_ports.values(), (200, HEALTHY_BODY)
    )
    http[lane_ports["dev"]] = (503, DEV_503_BODY)
    http.update(
        {
            "13002": (200, json.dumps({"status": "ok"})),
            # 8099 is kept green HERE ONLY, and only for the frozen as-deployed
            # fixture, which does still probe `deploy-agent-8099`. The RED-before
            # proof is "the 2026-07-30 artifact reported every endpoint it listed
            # as green while dev was down"; drop this entry and that script gets a
            # connection-refused 000 the outage being replayed never contained,
            # which would falsify the proof by fabrication rather than by fixing
            # anything. The current script no longer probes 8099 at all
            # (see test_phantom_deploy_agent_endpoint_is_not_probed), so this
            # entry is inert for every test that drives FIXED_SCRIPT.
            "8099": (200, json.dumps({"state": "idle"})),
            "3003": (200, "<html>ok</html>"),
            # PROD_RUNTIME_MAIN_PORT (28085) is kept green HERE ONLY, and only
            # for the frozen as-deployed fixture (OMN-18320). That fixture
            # hardcodes a literal `curl ... 127.0.0.1:28085/health` regardless
            # of the current policy/lane declarations -- it is a byte-for-byte
            # capture of the 2026-07-30 artifact and is never edited -- so it
            # always probes this port no matter what the current FIXED_SCRIPT
            # declares. `lane_ports` (and therefore this dict's
            # `dict.fromkeys(lane_ports.values(), ...)` line above) no longer
            # carries prod's port because FIXED_SCRIPT declares it
            # RUNTIME_LANE_UNPROBED; without this entry the as-deployed
            # fixture's still-live probe to 28085 would get a fabricated
            # connection-refused the replayed outage never contained, same
            # failure mode the 8099 comment above describes. FIXED_SCRIPT
            # itself never calls curl against this port at all post-OMN-18320,
            # so this entry is inert for every test that drives FIXED_SCRIPT.
            _policy_port("PROD_RUNTIME_MAIN_PORT"): (200, HEALTHY_BODY),
        }
    )
    return http


def _outage_docker() -> dict[str, Any]:
    return {
        "containers": [
            {
                "name": "omninode-runtime",
                "status": "Up 26 minutes (health: starting)",
                "started_at": "2026-07-30T16:19:00Z",
                "start_period_ns": 120 * 10**9,
            },
            {
                "name": "omnibase-infra-redpanda",
                "status": "Up 40 minutes (healthy)",
                "started_at": "2026-07-30T16:05:00Z",
            },
            {
                "name": "omnibase-infra-postgres",
                "status": "Up 40 minutes (healthy)",
                "started_at": "2026-07-30T16:05:00Z",
            },
            {
                "name": "omnibase-infra-valkey",
                "status": "Up 40 minutes (healthy)",
                "started_at": "2026-07-30T16:05:00Z",
            },
        ],
        "dangling": [],
    }


@pytest.fixture
def lane_ports() -> dict[str, str]:
    """The lanes the reporter probes, and their policy-resolved ports.

    Scoped to PROBED_LANE_PORT_KEYS rather than every policy lane (OMN-17150): a
    lane the script explicitly declares unprobed must not be asserted present in
    the digest. The parity test below is what keeps that declaration honest.
    """
    return {lane: _policy_port(key) for lane, key in PROBED_LANE_PORT_KEYS.items()}


# --------------------------------------------------------------------------
# AC 6 -- RED-before / GREEN-after against the artifact that actually runs.
# --------------------------------------------------------------------------


def test_as_deployed_reports_green_on_the_replayed_outage(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """RED-before: the live 2026-07-30 script never looked at the dev runtime."""
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_outage_http(lane_ports),
        docker_state=_outage_docker(),
    )
    report = _run(AS_DEPLOYED_SCRIPT, tmp_path, bin_dir)

    endpoints = report.split("*Runtime endpoints*", 1)[1].split("*Active issues*", 1)[0]
    # Endpoint labels are backtick-delimited, so match the whole label rather
    # than the bare port: "8085" is a substring of "runtime-18085".
    probed_labels = re.findall(r"- `([^`]+)`: HTTP", endpoints)
    assert probed_labels, endpoints
    assert not [
        label
        for label in probed_labels
        if re.search(rf"(?<!\d){lane_ports['dev']}$", label)
    ], (
        f"fixture is not the pre-fix artifact: it already probes the dev runtime ({probed_labels})"
    )
    # ...and every endpoint it does list is a green 200.
    assert "CRITICAL" not in endpoints and "WARNING" not in endpoints
    assert f"runtime-{lane_ports['stability-test']}`: HTTP 200 (OK)" in endpoints
    # The frozen as-deployed fixture hardcodes a literal probe against 28085
    # regardless of current lane declarations (OMN-18320) -- read the port
    # straight from the policy rather than through `lane_ports`, which now
    # excludes prod because FIXED_SCRIPT (not this fixture) declares it
    # unprobed.
    assert (
        f"runtime-{_policy_port('PROD_RUNTIME_MAIN_PORT')}`: HTTP 200 (OK)" in endpoints
    )
    # ...and `health: starting` never registered: the container_issues line is
    # OK and the whole report claims zero critical, zero warning.
    assert re.search(r"container_issues`: [^\n]*\(OK\)", report), report
    assert "Issues: *0 critical*, *0 warning*" in report, report
    assert "- No active warning/critical checks" in report, report


def test_fixed_reports_red_and_names_the_dev_runtime_on_the_same_state(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """GREEN-after: identical replayed state, fixed artifact, RED naming dev."""
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_outage_http(lane_ports),
        docker_state=_outage_docker(),
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)

    assert f"runtime-dev-{lane_ports['dev']}`: HTTP 503 (CRITICAL)" in report
    assert (
        f"runtime-stability-test-{lane_ports['stability-test']}`: HTTP 200 (OK)"
        in report
    )
    # OMN-18320: the fixed reporter no longer probes the retired prod lane at
    # all -- not green, not CRITICAL, absent. The as-deployed test above is
    # what still proves the historical artifact probed it; this is the
    # GREEN-after half of that same retirement.
    assert "runtime-prod-" not in report, report
    assert re.search(r"Issues: \*[1-9]\d* critical\*", report), report


# --------------------------------------------------------------------------
# AC 7 -- the omission cannot silently reappear.
# --------------------------------------------------------------------------


def test_every_lane_main_runtime_port_is_in_the_probe_set(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """Drop a lane from RUNTIME_LANE_SPECS and this fails."""
    http = dict.fromkeys(lane_ports.values(), (200, HEALTHY_BODY))
    http.update(
        {
            "13002": (200, '{"status":"ok"}'),
            "3003": (200, "ok"),
        }
    )
    bin_dir = _make_stub_bin(
        tmp_path, http=http, docker_state={"containers": [], "dangling": []}
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)

    endpoints = report.split("*Runtime endpoints*", 1)[1].split("*Active issues*", 1)[0]
    for lane, port in lane_ports.items():
        assert f"runtime-{lane}-{port}" in endpoints, (
            f"lane {lane} (:{port}) is not probed -- a lane's MAIN runtime health "
            f"endpoint was dropped from the alert (OMN-15509 AC2/AC7)"
        )
    # OMN-17150: and the converse. A lane declared unprobed must not show up in
    # the digest, or the declaration is a lie and the sandbox pages the on-call.
    for lane in UNPROBED_LANE_PORT_KEYS:
        assert f"runtime-{lane}-" not in endpoints, (
            f"lane {lane} is declared in RUNTIME_LANE_UNPROBED yet appears in the "
            f"digest -- either probe it deliberately or stop declaring it"
        )


def test_reporter_lane_specs_and_policy_lanes_are_in_two_way_parity() -> None:
    """RUNTIME_LANE_SPECS and runtime-policy.env must declare the same lane set.

    Two-way on purpose, because the two directions are different bugs:

    * a policy lane missing from the script is the OMN-15556 judge-lane blind
      spot -- a live runtime (:48085, seven containers on .201) whose death
      pages nobody while the digest keeps printing ``0 critical``;
    * a script lane missing from the policy is the inverse -- ``lane_main_port``
      can never resolve the key, so the reporter fails closed and alarms forever
      on a lane that does not exist.

    Neither direction is observable by iterating one hand-written map, which is
    how judge stayed invisible through OMN-15509 and OMN-15525.
    """
    script_lanes = _script_lane_specs(FIXED_SCRIPT)
    unprobed_lanes = _script_unprobed_lane_specs(FIXED_SCRIPT)

    overlap = sorted(set(script_lanes) & set(unprobed_lanes))
    assert not overlap, (
        f"lane(s) {overlap} appear in BOTH RUNTIME_LANE_SPECS and "
        f"RUNTIME_LANE_UNPROBED in {FIXED_SCRIPT.name} -- the tables must "
        f"partition the policy's lanes, not overlap"
    )

    declared = {**script_lanes, **unprobed_lanes}
    undeclared = sorted(set(LANE_PORT_KEYS) - set(declared))
    assert not undeclared, (
        f"lane(s) {undeclared} declare a *_RUNTIME_MAIN_PORT in "
        f"{RUNTIME_POLICY_ENV.name} but appear in neither RUNTIME_LANE_SPECS nor "
        f"RUNTIME_LANE_UNPROBED in {FIXED_SCRIPT.name} -- a live runtime lane "
        f"whose death pages nobody. Probe it, or declare it unprobed with a "
        f"reason; silence is not one of the options."
    )
    phantom = sorted(set(declared) - set(LANE_PORT_KEYS))
    assert not phantom, (
        f"lane(s) {phantom} appear in a reporter lane table but declare no "
        f"*_RUNTIME_MAIN_PORT in {RUNTIME_POLICY_ENV.name} -- the port can "
        f"never resolve, so the reporter alarms forever on a phantom lane"
    )
    assert declared == LANE_PORT_KEYS, (
        f"lane -> policy-key mapping disagrees between the reporter and the "
        f"policy: script={declared} policy={LANE_PORT_KEYS}"
    )

    # OMN-17150: the unprobed table is a narrow, declared exception -- never a
    # place a platform-owned lane can be parked. Moving any of these four out of
    # the probed table is the OMN-15556 blind spot with an extra step, so it
    # fails here regardless of what reason accompanies it.
    misfiled = sorted(MUST_BE_PROBED_LANES & set(unprobed_lanes))
    assert not misfiled, (
        f"lane(s) {misfiled} were moved into RUNTIME_LANE_UNPROBED. These lanes "
        f"are platform-owned and must be probed; only a lane nobody promised to "
        f"keep up (a collaborator sandbox) belongs in that table."
    )
    missing_required = sorted(MUST_BE_PROBED_LANES - set(script_lanes))
    assert not missing_required, (
        f"lane(s) {missing_required} are missing from RUNTIME_LANE_SPECS entirely"
    )


def test_lane_specs_are_sourced_from_the_rendered_runtime_policy() -> None:
    """The map is config-driven; hardcoding a port per call site regresses AC2."""
    body = FIXED_SCRIPT.read_text()
    for lane, key in LANE_PORT_KEYS.items():
        assert f"{lane}|{key}" in body, f"lane {lane} not declared against {key}"
    assert "policy_env_value" in body


def test_lane_specs_carry_no_hardcoded_fallback_ports() -> None:
    """OMN-15525: the spec table must not smuggle literal ports back in.

    The OMN-15509 revision declared ``dev|DEV_RUNTIME_MAIN_PORT|8085`` and
    substituted that literal whenever the policy lookup came back empty, so a
    renamed key or an unrendered policy file silently probed a guessed port.
    """
    source = FIXED_SCRIPT.read_text()
    tables = {
        "RUNTIME_LANE_SPECS": re.search(
            r"RUNTIME_LANE_SPECS=\((.*?)\n\)", source, re.DOTALL
        ),
    }
    # OMN-17150: the unprobed table carries the same lane|key rows and must obey
    # the same rule -- a literal port smuggled in there would resurface the
    # moment the lane graduates back into the probed table.
    if "RUNTIME_LANE_UNPROBED=(" in source:
        tables["RUNTIME_LANE_UNPROBED"] = re.search(
            r"RUNTIME_LANE_UNPROBED=\((.*?)\n\)", source, re.DOTALL
        )

    assert tables["RUNTIME_LANE_SPECS"], "RUNTIME_LANE_SPECS table not found"
    for table_name, table in tables.items():
        assert table, f"{table_name} table not found"
        for raw in table.group(1).strip().splitlines():
            entry = raw.strip().strip('"')
            if not entry or entry.startswith("#"):
                continue
            fields = entry.split("|")
            assert len(fields) == 2, (
                f"{table_name} row {entry!r} carries more than lane|key -- a "
                "third field is the hardcoded fallback port OMN-15525 removed"
            )
            assert not re.fullmatch(r"\d+", fields[1]), entry


def test_cron_unit_points_at_the_versioned_script_name() -> None:
    unit = CRON_UNIT.read_text()
    assert "omninode-system-slack-report.sh" in unit
    assert "--mode alert" in unit and "--mode digest" in unit


# --------------------------------------------------------------------------
# OMN-16789 follow-up -- the phantom `deploy-agent-8099` probe is gone.
# --------------------------------------------------------------------------


def test_phantom_deploy_agent_endpoint_is_not_probed(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """`deploy-agent-8099` named a service that does not exist on `.201`.

    Verified on the host 2026-08-27 after the OMN-16789 install: no listener on
    8099 (`ss -lntp`), no container publishing it (`docker ps -a`), and
    `curl :8099/health` -> 000. The real deploy runner
    (`omninode-deploy-runner`, healthy, image `omninode-runner:latest`)
    publishes NO ports at all -- `NetworkSettings.Ports` is `{}` -- because it
    is HMAC-command driven (`DEPLOY_AGENT_HMAC_SECRET`), not an HTTP service.
    Port 8099 is in fact allocated by the service catalog to a different,
    profile-gated fixture: `docker/catalog/services/fault-inject-fixture.yaml`
    declares `ports.external: 8099`.

    So the probe could never be anything but `000`. Fail-closed on 000 is
    correct for a real endpoint and stays; the defect is that this endpoint was
    never real. It was one of only two standing criticals on the host and,
    per OMN-16789's own measurement, `CRITICAL` on all 39 ticks -- a permanent
    false-RED, which is the "crying wolf" failure direction this script's
    header calls fatal to a monitor.

    The stub below deliberately omits an "8099" entry, so it answers
    connection-refused there. Restore the probe and this test fails RED.
    """
    http = dict.fromkeys(lane_ports.values(), (200, HEALTHY_BODY))
    http.update(
        {
            "13002": (200, '{"status":"ok"}'),
            "3003": (200, "ok"),
        }
    )
    bin_dir = _make_stub_bin(
        tmp_path, http=http, docker_state={"containers": [], "dangling": []}
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "deploy-agent" not in report, (
        "the phantom deploy-agent endpoint is still probed:\n" + report
    )
    assert "8099" not in report, "8099 is still referenced in the report:\n" + report
    # And removing it must not have silenced the rest: a fully green fleet
    # still reports zero criticals rather than nothing at all.
    assert "*0 critical*" in report, report


# --------------------------------------------------------------------------
# AC 3 -- 200 with a non-healthy body is RED.
# --------------------------------------------------------------------------


def test_http_200_with_unhealthy_body_is_critical(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    http = dict.fromkeys(lane_ports.values(), (200, HEALTHY_BODY))
    # 200 but the body says otherwise -- the exact case substring matching missed.
    http[lane_ports["dev"]] = (
        200,
        json.dumps({"status": "degraded", "healthy": False}),
    )
    http.update(
        {
            "13002": (200, '{"status":"ok"}'),
            "3003": (200, "ok"),
        }
    )
    bin_dir = _make_stub_bin(
        tmp_path, http=http, docker_state={"containers": [], "dangling": []}
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert f"runtime-dev-{lane_ports['dev']}`: HTTP 200 (CRITICAL)" in report


def test_http_200_with_unresolvable_body_fails_closed(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    http = dict.fromkeys(lane_ports.values(), (200, HEALTHY_BODY))
    http[lane_ports["dev"]] = (200, "OK")  # not JSON, no resolvable status
    http.update(
        {
            "13002": (200, '{"status":"ok"}'),
            "3003": (200, "ok"),
        }
    )
    bin_dir = _make_stub_bin(
        tmp_path, http=http, docker_state={"containers": [], "dangling": []}
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert f"runtime-dev-{lane_ports['dev']}`: HTTP 200 (CRITICAL)" in report


# --------------------------------------------------------------------------
# AC 5 -- an endpoint that cannot be probed is RED, never omitted.
# --------------------------------------------------------------------------


def test_unreachable_runtime_endpoint_is_critical_not_skipped(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    http = dict.fromkeys(lane_ports.values(), (200, HEALTHY_BODY))
    del http[lane_ports["dev"]]  # connection refused
    http.update(
        {
            "13002": (200, '{"status":"ok"}'),
            "3003": (200, "ok"),
        }
    )
    bin_dir = _make_stub_bin(
        tmp_path, http=http, docker_state={"containers": [], "dangling": []}
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert f"runtime-dev-{lane_ports['dev']}`: HTTP 000 (CRITICAL)" in report


# --------------------------------------------------------------------------
# AC 4 -- `health: starting` past start_period alarms; AC 5 -- Exit(0) does not.
# --------------------------------------------------------------------------


def _all_green_http(lane_ports: dict[str, str]) -> dict[str, tuple[int, str]]:
    http = dict.fromkeys(lane_ports.values(), (200, HEALTHY_BODY))
    http.update(
        {
            "13002": (200, '{"status":"ok"}'),
            "3003": (200, "ok"),
        }
    )
    return http


def test_container_starting_past_start_period_is_critical(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    docker_state = {
        "containers": [
            {
                "name": "omninode-runtime",
                "status": "Up 26 minutes (health: starting)",
                "started_at": "2026-07-30T16:19:00Z",
                "start_period_ns": 120 * 10**9,
            }
        ],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "starting_past_start_period=omninode-runtime" in report
    assert "container_issues" in report
    assert re.search(r"Issues: \*[1-9]\d* critical\*", report), report


def test_container_still_inside_start_period_does_not_alarm(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """A genuinely-still-booting container is not a page."""
    from datetime import UTC, datetime

    docker_state = {
        "containers": [
            {
                "name": "omninode-runtime",
                "status": "Up 3 seconds (health: starting)",
                "started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "start_period_ns": 600 * 10**9,
            }
        ],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "starting_past_start_period=none" in report
    assert "Issues: *0 critical*" in report


def test_expected_exit_zero_oneshot_does_not_alarm(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    docker_state = {
        "containers": [
            {
                "name": "omnibase-infra-migration",
                "status": "Exited (0) 4 minutes ago",
                "started_at": "2026-07-30T16:05:00Z",
            }
        ],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "exited_nonzero=none" in report
    assert "Issues: *0 critical*" in report


def test_nonzero_exit_alarms(tmp_path: Path, lane_ports: dict[str, str]) -> None:
    docker_state = {
        "containers": [
            {
                "name": "omninode-runtime-effects",
                "status": "Exited (1) 2 minutes ago",
                "started_at": "2026-07-30T16:05:00Z",
            }
        ],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "exited_nonzero=omninode-runtime-effects" in report
    assert re.search(r"Issues: \*[1-9]\d* critical\*", report), report


# --------------------------------------------------------------------------
# OMN-18571 -- `created` containers are classified by age and ownership.
#
# The defect: `created` was a single grep count with no age floor and no
# ownership distinction, and ANY nonzero value set docker_status=CRITICAL. Two
# measured false-REDs followed on .201 on 2026-09-17 (census report
# lab-hygiene-census-1054-20260917T1057Z.md section 1):
#
#   (a) compose legitimately holds containers in `created` for seconds while a
#       `depends_on: service_healthy` dependency boots, so the check reddened on
#       every dev-lane redeploy -- the delivery chain working reported as an
#       outage;
#   (b) 48 unowned CI/testcontainers leftovers aged 1-3 days held the host at
#       CRITICAL permanently with zero unhealthy/restarting/dead, which is the
#       "crying wolf" direction this file's own header calls fatal to a monitor.
#
# What did NOT change, and is pinned by test_unhealthy/restarting/dead below
# plus the three pre-existing exit/start-period tests above: `unhealthy`,
# `restarting`, `dead`, a container past its own declared start_period, a
# non-zero exit, and a failed docker query all still drive CRITICAL.
# --------------------------------------------------------------------------


def _iso_ago(seconds: int) -> str:
    """A Docker-shaped `.Created` timestamp `seconds` in the past.

    Nanosecond precision on purpose: that is what `docker inspect` emits, and
    the script's own `epoch_from_iso` has to trim it. A test that fed whole
    seconds would not exercise the trim.
    """
    from datetime import UTC, datetime, timedelta

    stamp = datetime.now(UTC) - timedelta(seconds=seconds)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S") + ".123456789Z"


def _created(
    name: str,
    *,
    age_s: int | None = None,
    created_at: str | None = None,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One container sitting in Docker `created` status."""
    return {
        "name": name,
        "status": "Created",
        "started_at": "0001-01-01T00:00:00Z",
        "created_at": _iso_ago(age_s) if created_at is None else created_at,
        "labels": labels or {},
    }


def test_fresh_compose_held_container_does_not_trip_critical(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC1 -- the in-flight redeploy case, measured live on 2026-09-17T06:52:39.

    Three containers of the `omnibase-infra` project sat in `created` seconds
    before the census probe because compose was holding them on
    `omninode-runtime:service_healthy`. That is a deploy in progress, not a
    fault, and it must not page.
    """
    docker_state = {
        "containers": [
            _created(
                name,
                age_s=5,
                labels={"com.docker.compose.project": "omnibase-infra"},
            )
            for name in (
                "omninode-runtime-effects",
                "omnibase-infra-runtime-worker-1",
                "omninode-contract-resolver",
            )
        ],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "Issues: *0 critical*" in report, report
    # The raw count is still reported -- it is the classification that changed.
    assert "created=3" in report, report
    assert "created_fresh=3" in report, report
    assert "unowned_debris=0" in report, report


def test_aged_compose_held_container_is_a_warning_naming_its_project(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC2 -- a compose project still holding a container two hours later is a
    real stall worth saying out loud, but it is a WARNING: the project is named
    and there is an owner to name it to, so it is not the unattributable debris
    case and it does not page."""
    docker_state = {
        "containers": [
            _created(
                "omnibase-infra-stability-test-runtime-worker-1",
                age_s=7200,
                labels={"com.docker.compose.project": "omnibase-infra-stability-test"},
            )
        ],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "created_pending=omnibase-infra-stability-test:1" in report, report
    assert "Issues: *0 critical*" in report, report
    assert re.search(r"- WARNING `container_issues`:", report), report


def test_aged_unowned_container_is_warning_unowned_debris(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC3 -- the chronic baseline: a crashed testcontainers session leaves both
    the workload container and its own Ryuk reaper in `created` forever, with no
    compose project label and therefore no lane that owns them."""
    docker_state = {
        "containers": [
            _created("testcontainers-ryuk-8f21", age_s=172800),
            _created("repo-scripts-db-pg-34990041118-1", age_s=140000),
        ],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "unowned_debris=2" in report, report
    assert "Issues: *0 critical*" in report, report
    assert re.search(r"- WARNING `container_issues`:", report), report


def test_retention_labelled_container_is_listed_and_never_counted(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC4 -- the live keycloak case, held deliberately for OMN-18366's own
    closer. Counting a container somebody labelled `keep this` as a problem
    argues for deleting it, which is the opposite of what the label says. It is
    named in the report so the retention stays visible, and counted nowhere."""
    docker_state = {
        "containers": [
            _created(
                "c9-keycloak-retained",
                age_s=172800,
                labels={
                    "onex.cleanup-owner": "close_18354",
                    "onex.purpose": "c9-keycloak-image-retention",
                    "onex.ticket": "OMN-18366",
                },
            )
        ],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "created_retained=c9-keycloak-retained" in report, report
    assert "unowned_debris=0" in report, report
    assert "created_pending=none" in report, report
    assert "Issues: *0 critical*" in report, report


def test_cleanup_owner_alone_is_enough_to_retain(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC4, second shape -- either label retains on its own. A container whose
    `onex.purpose` does not end `-retention` but which names a cleanup owner is
    still somebody's, so it is not unattributed debris."""
    docker_state = {
        "containers": [
            _created(
                "held-by-a-named-owner",
                age_s=172800,
                labels={"onex.cleanup-owner": "close_18354"},
            ),
            _created(
                "held-by-purpose-only",
                age_s=172800,
                labels={"onex.purpose": "omn18366-image-retention"},
            ),
        ],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "unowned_debris=0" in report, report
    assert "held-by-a-named-owner" in report, report
    assert "held-by-purpose-only" in report, report
    assert "Issues: *0 critical*" in report, report


def test_a_purpose_that_merely_mentions_retention_does_not_retain(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC4, negative control -- the match is on the END of the purpose value,
    not a substring anywhere in it (CLAUDE.md rule 15). A purpose of
    `retention-policy-probe` describes a test OF retention, not a container
    somebody asked to keep, and it must still count as debris. Without this the
    exclusion is a hole any label containing the word can walk through."""
    docker_state = {
        "containers": [
            _created(
                "retention-policy-probe-1",
                age_s=172800,
                labels={"onex.purpose": "retention-policy-probe"},
            )
        ],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "unowned_debris=1" in report, report
    assert "created_retained=none" in report, report


def test_unageable_created_container_counts_as_debris_rather_than_excused(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC6 -- the fail-closed direction, and the one an implementation is most
    likely to get backwards.

    A container whose creation timestamp cannot be parsed cannot be proven
    YOUNGER than the floor. The floor is an excuse for not counting something,
    so an unprovable age must not earn it: the container is counted. An
    implementation that treats "cannot age" as "too young to count" turns every
    unparseable timestamp into silence, which is the exact shape of false green
    this file's header refuses.
    """
    docker_state = {
        "containers": [_created("age-unknown-debris", created_at="not-a-timestamp")],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "unowned_debris=1" in report, report
    assert "created_fresh=0" in report, report


def test_the_created_age_floor_is_configurable(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC1 -- the floor is a declared, env-overridable config entry, not a
    literal welded into the decision path. Positive control on the same state:
    the SAME container reads as debris at the default floor and as fresh at a
    floor raised above its age, so the test cannot pass against an
    implementation that ignores the setting."""
    docker_state = {
        "containers": [_created("aged-out", age_s=3600)],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    default_floor = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "unowned_debris=1" in default_floor, default_floor

    raised = _run(
        FIXED_SCRIPT,
        tmp_path,
        bin_dir,
        extra_env={"OMNINODE_ALERT_CREATED_FLOOR_SECONDS": "7200"},
    )
    assert "unowned_debris=0" in raised, raised
    assert "created_fresh=1" in raised, raised
    assert "created_floor_seconds=7200" in raised, raised


def test_created_floor_is_declared_in_the_config_block(tmp_path: Path) -> None:
    """AC1, second falsifier -- the floor is resolved once, in the config block,
    with the same `${OMNINODE_ALERT_*:-<default>}` shape as every other tunable
    in this file. A magic number inside the classification pipeline would pass
    the behavioural tests above and still be the defect OMN-16789's header
    warns about: a literal in the decision path that nobody can find."""
    body = FIXED_SCRIPT.read_text()
    assert re.search(
        r"^CREATED_FLOOR_SECONDS=\$\{OMNINODE_ALERT_CREATED_FLOOR_SECONDS:-600\}",
        body,
        flags=re.MULTILINE,
    ), "the created age floor must be declared in the config block with a default"
    # Anchored on the DEFINITION line, not the first mention of the name: the
    # config entry's own comment cites the function, and splitting on the bare
    # name would scan that comment instead of the classifier body.
    classifier = body.split("\nclassify_created_containers() {\n", 1)
    assert len(classifier) == 2, "classify_created_containers() must be defined"
    decision_path = classifier[1].split("\n}\n", 1)[0]
    assert "600" not in decision_path, (
        "the floor's default must not be repeated as a literal inside the "
        f"classifier: {decision_path}"
    )


@pytest.mark.parametrize(
    ("name", "status"),
    [
        ("omninode-runtime", "Up 3 minutes (unhealthy)"),
        ("omninode-runtime-effects", "Restarting (1) 5 seconds ago"),
        ("omnibase-infra-postgres", "Dead"),
    ],
)
def test_real_container_faults_still_drive_critical(
    tmp_path: Path, lane_ports: dict[str, str], name: str, status: str
) -> None:
    """AC5 -- the conditions that were always worth paging on are untouched.

    This change narrows exactly one input (`created`). If it widened into the
    OMN-15509 fix it sits beside, these three plus the three pre-existing
    exit/start-period tests above are what fails.
    """
    docker_state = {
        "containers": [
            {
                "name": name,
                "status": status,
                "started_at": "2026-07-30T16:05:00Z",
            }
        ],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert re.search(r"Issues: \*[1-9]\d* critical\*", report), report


def test_an_unhealthy_container_pages_even_beside_fresh_created_ones(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC5 -- the mixed state. A redeploy in flight (fresh compose-held
    containers) alongside a genuinely unhealthy container still pages: the
    floor silences the `created` input, never the row."""
    docker_state = {
        "containers": [
            _created(
                "omninode-runtime-effects",
                age_s=5,
                labels={"com.docker.compose.project": "omnibase-infra"},
            ),
            {
                "name": "omninode-runtime",
                "status": "Up 3 minutes (unhealthy)",
                "started_at": "2026-07-30T16:05:00Z",
            },
        ],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "created_fresh=1" in report, report
    assert re.search(r"Issues: \*[1-9]\d* critical\*", report), report


def test_container_issues_keeps_every_pre_existing_output_key(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC7 -- extend, never rename. The Slack renderer reads this row and so
    could anything downstream, so the six keys that were there before this
    change are still there, with the same meanings. `created=` in particular
    stays the RAW total of created-status containers -- the classification is
    reported in the new fields beside it, not by quietly redefining an existing
    one, which would read as a fix to anyone diffing the output."""
    docker_state = {
        "containers": [
            _created("fresh-one", age_s=5),
            _created("aged-one", age_s=172800),
        ],
        "dangling": [],
    }
    bin_dir = _make_stub_bin(
        tmp_path, http=_all_green_http(lane_ports), docker_state=docker_state
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    for key in (
        "unhealthy=",
        "restarting=",
        "dead=",
        "created=",
        "starting_past_start_period=",
        "exited_nonzero=",
    ):
        assert key in report, f"{key} disappeared from container_issues: {report}"
    assert "created=2" in report, report
    for key in (
        "created_floor_seconds=",
        "created_fresh=",
        "created_pending=",
        "unowned_debris=",
        "created_retained=",
    ):
        assert key in report, f"{key} missing from container_issues: {report}"


def test_a_failed_created_query_is_still_critical(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC6 -- fail-closed on the query itself, unchanged from before. A docker
    call that did not run is not evidence that nothing is wrong, and the
    classifier must not convert that into a quiet WARNING."""
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_all_green_http(lane_ports),
        docker_state={"containers": [], "dangling": []},
    )
    # Replace the docker stub with one that fails every call.
    _write(
        bin_dir / "docker",
        "#!/usr/bin/env bash\necho 'docker: cannot connect' >&2\nexit 1\n",
        executable=True,
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "docker_query=FAILED" in report, report
    assert re.search(r"Issues: \*[1-9]\d* critical\*", report), report


def test_the_measured_census_state_no_longer_pages(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """The 2026-09-17T10:57Z census state, replayed in proportion.

    51 created containers: 3 compose-held and seconds old, 47 unowned and 1-3
    days old, 1 retention-labelled. Zero unhealthy, zero restarting, zero dead.
    That state held the host at CRITICAL continuously. It is now one WARNING
    row that names the debris count, which is a thing somebody can act on
    rather than a colour nobody believes.
    """
    containers: list[dict[str, Any]] = [
        _created(name, age_s=5, labels={"com.docker.compose.project": "omnibase-infra"})
        for name in (
            "omninode-runtime-effects",
            "omnibase-infra-runtime-worker-1",
            "omninode-contract-resolver",
        )
    ]
    containers += [
        _created(f"onex-egress-probe-{index}", age_s=172800) for index in range(32)
    ]
    containers += [
        _created(f"testcontainers-ryuk-{index}", age_s=140000) for index in range(9)
    ]
    containers += [_created(f"stray-{index}", age_s=90000) for index in range(6)]
    containers.append(
        _created(
            "c9-keycloak-retained",
            age_s=172800,
            labels={
                "onex.cleanup-owner": "close_18354",
                "onex.purpose": "c9-keycloak-image-retention",
            },
        )
    )
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_all_green_http(lane_ports),
        docker_state={"containers": containers, "dangling": []},
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)
    assert "created=51" in report, report
    assert "created_fresh=3" in report, report
    assert "unowned_debris=47" in report, report
    assert "created_retained=c9-keycloak-retained" in report, report
    assert "Issues: *0 critical*" in report, report
    assert re.search(r"- WARNING `container_issues`:", report), report


# --------------------------------------------------------------------------
# Prod stays read-only.
# --------------------------------------------------------------------------


def test_prod_lane_is_probed_with_a_plain_get_only() -> None:
    body = FIXED_SCRIPT.read_text()
    assert re.search(r"curl .*-X GET .*/health", body), (
        "runtime probe must be an explicit GET"
    )
    for verb in ("-X POST", "-X PUT", "-X DELETE", "-X PATCH"):
        assert verb not in body, f"reporter must never issue {verb}"


# --------------------------------------------------------------------------
# OMN-18320 -- the lab compose prod lane was retired 2026-09-13; the reporter
# must stop probing it rather than page a real-but-meaningless CRITICAL for a
# lane nobody is bringing back.
# --------------------------------------------------------------------------


def test_prod_lane_is_declared_unprobed_after_retirement() -> None:
    """RED-before this PR: `prod` sat in RUNTIME_LANE_SPECS (the probed table)
    with no corresponding compose project on .201 any more, so every /15 tick
    reported a fresh CRITICAL HTTP 000 for `runtime-prod-28085` -- a real fact
    about a lane that was deliberately shut down, not an incident.
    """
    unprobed_lanes = _script_unprobed_lane_specs(FIXED_SCRIPT)
    assert "prod" in unprobed_lanes, (
        "prod must be declared in RUNTIME_LANE_UNPROBED now that the "
        "omnibase-infra-prod compose lane no longer exists (OMN-18320) -- "
        f"found: {unprobed_lanes}"
    )
    probed_lanes = _script_lane_specs(FIXED_SCRIPT)
    assert "prod" not in probed_lanes, (
        "prod must not remain in RUNTIME_LANE_SPECS -- the compose lane it "
        f"names was shut down 2026-09-13 (OMN-18320): {probed_lanes}"
    )


def test_retired_prod_lane_never_receives_a_probe_and_produces_no_critical(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """The direct reproduction of the OMN-18320 defect against a live-shaped
    fleet: every lane the reporter is still responsible for (dev,
    stability-test, judge) is healthy, and nothing stubs a response for
    PROD_RUNTIME_MAIN_PORT at all -- exactly like the real .201 host today,
    where nothing listens on :28085. Before this PR's fix that produced a
    connection-refused CRITICAL for a lane nobody is bringing back; after it,
    the port is never dialed and the fleet reads clean.
    """
    prod_port = _policy_port("PROD_RUNTIME_MAIN_PORT")
    assert prod_port not in lane_ports.values(), (
        "test setup error: prod's port must not be one of the reporter's own "
        "probed lanes for this to be a meaningful reproduction"
    )
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_all_green_http(lane_ports),  # deliberately no entry for prod_port
        docker_state={"containers": [], "dangling": []},
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)

    assert f"runtime-prod-{prod_port}" not in report, (
        f"the retired prod lane was probed and appears in the report:\n{report}"
    )
    assert "CRITICAL" not in report, (
        f"a lane nobody stubbed (prod, :{prod_port}) produced a CRITICAL even "
        f"though it was never declared probed:\n{report}"
    )
    assert "Issues: *0 critical*, *0 warning*" in report, report


def test_positive_control_platform_lanes_still_probed_after_prod_retirement(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """Positive control for the test above: dev, stability-test and judge --
    the lanes the platform is actually responsible for -- still render as
    probed endpoints. Without this, a reporter that probed NOTHING would also
    pass the retirement test above vacuously.
    """
    assert set(lane_ports) == {"dev", "stability-test", "judge"}, (
        f"expected exactly the three platform-owned lanes to be probed, got "
        f"{sorted(lane_ports)} -- either a lane was dropped or a new one "
        f"needs a decision, not a silent default"
    )
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_all_green_http(lane_ports),
        docker_state={"containers": [], "dangling": []},
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)

    for lane, port in lane_ports.items():
        assert f"runtime-{lane}-{port}`: HTTP 200 (OK)" in report, (
            f"positive control failed: lane {lane} (:{port}) did not render as "
            f"a healthy probed endpoint:\n{report}"
        )
    assert "Issues: *0 critical*, *0 warning*" in report, report


# --------------------------------------------------------------------------
# OMN-15525 -- the two false-green/false-RED defects found by DEPLOYING the
# OMN-15509 fix to .201 (installed 18:12Z, rolled back 18:13:29Z after it
# reported CRITICAL on all three demonstrably healthy lanes).
# --------------------------------------------------------------------------


def test_healthy_lane_with_a_realistic_body_is_not_critical(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """A 200 + healthy body larger than the display excerpt must read OK.

    RED-before: `check_runtime_lane` truncated the body to 180 bytes and then
    parsed THAT with jq. A real .201 body is 2644 bytes, so jq always failed
    ("Unfinished string at EOF"), the verdict was always "unresolvable", and
    fail-closed reported CRITICAL for every lane on a healthy fleet.
    """
    assert len(HEALTHY_BODY) > 180, "fixture must cross the excerpt boundary"
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_all_green_http(lane_ports),
        docker_state={"containers": [], "dangling": []},
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)

    for lane, port in lane_ports.items():
        assert f"runtime-{lane}-{port}`: HTTP 200 (OK)" in report, (
            f"lane {lane} went non-OK on a healthy 200 whose body merely exceeds "
            f"the display excerpt -- the body is being parsed after truncation "
            f"(OMN-15525):\n{report}"
        )
    assert "could not be resolved from body" not in report, report
    assert "Issues: *0 critical*, *0 warning*" in report, report


def test_healthy_body_fixture_is_the_unmodified_201_capture() -> None:
    """Provenance guard for the OMN-15547 replay artifact.

    The value of this fixture is that nobody wrote it. If it is edited it
    becomes a reconstruction again -- indistinguishable, to every other test in
    this file, from the hand-typed literal it replaced -- so the edit has to
    fail here rather than quietly weaken every assertion downstream.
    """
    digest = hashlib.sha256(HEALTHY_BODY_FIXTURE.read_bytes()).hexdigest()
    assert digest == HEALTHY_BODY_SHA256, (
        "the healthy-body fixture no longer matches the response captured from "
        "GET http://127.0.0.1:8085/health on .201 at 2026-07-31T00:03:06Z; "
        "re-capture it rather than editing it"
    )
    # The property the byte count is standing in for: the real payload is an
    # order of magnitude past the reporter's 180-byte pre-parse cut. A fixture
    # that drifts back under that boundary cannot see the OMN-15525 class at all.
    assert len(HEALTHY_BODY_FIXTURE.read_bytes()) > 2000, (
        "a real .201 /health body is 2079-2644 bytes; a fixture materially "
        "smaller than that is not the thing the reporter parses in production"
    )
    # It must also still be the payload the reporter's jq expression reads.
    parsed = json.loads(HEALTHY_BODY)
    assert parsed["details"]["healthy"] is True, (
        "the capture must be a HEALTHY body -- this case replays the false-RED "
        "direction (a healthy fleet paged CRITICAL), so a body that is genuinely "
        "unhealthy would make every assertion built on it vacuous"
    )


def test_verdict_is_independent_of_the_display_excerpt_size(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """Shrinking the reported excerpt must not change any lane's status.

    This is the invariant the defect violated: display truncation is cosmetic,
    so driving it to an absurdly small value must leave every verdict intact.
    """
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_all_green_http(lane_ports),
        docker_state={"containers": [], "dangling": []},
    )
    baseline = _run(FIXED_SCRIPT, tmp_path, bin_dir)

    tiny_path = tmp_path / "tiny"
    tiny_path.mkdir()
    tiny_bin = _make_stub_bin(
        tiny_path,
        http=_all_green_http(lane_ports),
        docker_state={"containers": [], "dangling": []},
    )
    tiny = _run(
        FIXED_SCRIPT,
        tiny_path,
        tiny_bin,
        extra_env={"OMNINODE_ALERT_BODY_EXCERPT_BYTES": "12"},
    )

    def _statuses(report: str) -> list[tuple[str, str]]:
        section = report.split("*Runtime endpoints*", 1)[1].split("*Active issues*", 1)[
            0
        ]
        return re.findall(r"- `([^`]+)`: HTTP \d+ \((\w+)\)", section)

    assert _statuses(baseline) == _statuses(tiny), (
        f"verdicts moved when only the display excerpt changed:\n"
        f"{_statuses(baseline)}\nvs\n{_statuses(tiny)}"
    )
    # Guard against a vacuous pass: "all CRITICAL == all CRITICAL" also
    # satisfies the equality above, and that is precisely the broken state.
    # The fleet here is healthy, so every verdict must be OK.
    assert _statuses(baseline), baseline
    assert all(status == "OK" for _, status in _statuses(baseline)), (
        f"stability check is vacuous -- the healthy baseline is not all OK:\n{baseline}"
    )


def test_missing_runtime_policy_file_is_critical_not_a_hardcoded_port(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """Rule 8: an absent policy file must alarm, not fall back to 8085/18085/28085.

    RED-before: `policy_env_value` returned success on a missing file and
    `lane_main_port` substituted the literal port, so the reporter kept probing
    guessed ports and reported green with no indication the lane map had
    stopped resolving.
    """
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_all_green_http(lane_ports),
        docker_state={"containers": [], "dangling": []},
    )
    report = _run(
        FIXED_SCRIPT,
        tmp_path,
        bin_dir,
        extra_env={"OMNINODE_RUNTIME_POLICY_ENV": str(tmp_path / "no-such-policy.env")},
    )

    for lane in PROBED_LANE_PORT_KEYS:
        assert f"runtime-{lane}-unresolved" in report, (
            f"lane {lane} did not report an unresolvable port with the policy "
            f"file absent -- it fell back to a hardcoded port (OMN-15525):\n{report}"
        )
    for lane, port in lane_ports.items():
        assert f"runtime-{lane}-{port}" not in report, (
            f"lane {lane} probed literal :{port} with no policy file present"
        )
    assert re.search(r"Issues: \*[1-9]\d* critical\*", report), report


def test_renamed_policy_key_is_critical(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """A policy file that exists but no longer carries the key must alarm."""
    partial = tmp_path / "partial-policy.env"
    # Generated from the derived lane map (OMN-15556): dev's key is renamed out
    # from under the reporter, every other DECLARED lane still resolves.
    # Hardcoding three lines here meant a newly-declared lane was silently
    # absent from the partial policy, so it read as unresolved for the wrong
    # reason and the test proved nothing about that lane.
    #
    # Scoped to the PROBED lanes (OMN-17150): a lane the reporter never looks up
    # cannot be renamed out from under it, and writing a key it never reads
    # would prove nothing either way.
    partial.write_text(
        "".join(
            f"{key}_RENAMED={lane_ports[lane]}\n"
            if lane == "dev"
            else f"{key}={lane_ports[lane]}\n"
            for lane, key in PROBED_LANE_PORT_KEYS.items()
        )
    )
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_all_green_http(lane_ports),
        docker_state={"containers": [], "dangling": []},
    )
    report = _run(
        FIXED_SCRIPT,
        tmp_path,
        bin_dir,
        extra_env={"OMNINODE_RUNTIME_POLICY_ENV": str(partial)},
    )

    assert "runtime-dev-unresolved" in report, report
    assert f"runtime-dev-{lane_ports['dev']}" not in report, report
    # The lanes whose keys still resolve are unaffected.
    assert (
        f"runtime-stability-test-{lane_ports['stability-test']}`: HTTP 200 (OK)"
        in report
    ), report
    assert re.search(r"Issues: \*[1-9]\d* critical\*", report), report


# --------------------------------------------------------------------------
# OMN-15525 -- `--mode alert` is the path that actually pages, and it had NO
# behavioural coverage at all. Both prior revisions rendered a CRITICAL lane
# into the digest TEXT while computing an EMPTY `$issues`, so the alert branch
# took "clean" and posted nothing. Fixing the probe (OMN-15509) and the
# truncation is worthless if the alert still cannot fire.
# --------------------------------------------------------------------------

_SLACK_CURL = """#!/usr/bin/env bash
# Slack-aware curl: records chat.postMessage payloads, delegates everything
# else to the real stub so endpoint probing is unchanged.
is_slack=0
payload=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[$i]}" in
    https://slack.com/*) is_slack=1 ;;
    -d) payload="${args[$((i+1))]}" ;;
  esac
done
if (( is_slack )); then
  # One file per post: the payload is multi-line JSON (jq -n pretty-prints), so
  # appending to a shared file would not round-trip.
  mkdir -p "__POSTS__"
  printf '%s' "$payload" > "__POSTS__/$(date +%s%N)-$$.json"
  printf '%s' '{"ok":true}'
  exit 0
fi
exec "__REALCURL__" "$@"
"""


def _run_alert(
    script: Path,
    tmp_path: Path,
    bin_dir: Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Drive ``script`` in ``--mode alert``; return (log text, Slack post texts)."""
    posts = tmp_path / "slack-posts"
    real_curl = bin_dir / "curl-http"
    (bin_dir / "curl").rename(real_curl)
    _write(
        bin_dir / "curl",
        _SLACK_CURL.replace("__POSTS__", str(posts)).replace(
            "__REALCURL__", str(real_curl)
        ),
        executable=True,
    )

    env = dict(os.environ)
    env.update(
        {
            "OMNINODE_ALERT_ENV_FILE": str(tmp_path / "absent.env"),
            "OMNINODE_INFRA_REPO_ROOT": str(REPO_ROOT),
            "OMNINODE_RUNTIME_POLICY_ENV": str(RUNTIME_POLICY_ENV),
            "SLACK_BOT_TOKEN": "test-token",
            "SLACK_CHANNEL_ID": "C-TEST",
            # OMN-15550: the reporter now shells out to the required-context
            # probe from collect(). These lane tests are about disk/docker/
            # endpoint classification and must not acquire a GitHub network
            # dependency; the probe's own rows are asserted in
            # test_omninode_ci_required_context_probe.py.
            "OMNINODE_CI_PROBE_ENABLED": "0",
            # OMN-18567: same reasoning one row up. The reporter also reads
            # the runner-tree converge verdict from collect(). These tests
            # are about disk/docker/endpoint classification and must not
            # acquire a dependency on a state file that a cron tick writes;
            # the collector's own rows are asserted in the OMN-18567 block
            # at the end of this file, which switches it back on.
            "OMNINODE_RUNNER_TREE_CHECK_ENABLED": "0",
            # OMN-18944: same reasoning again. The reporter also reads the
            # backup freshness gate's latest run from collect(). These tests
            # must not acquire a GitHub network dependency; that check's own
            # rows are asserted in
            # tests/scripts/test_postgres_backup_freshness_row_omn18944.py,
            # which drives it through its fetch seam against recorded payloads.
            "OMNINODE_BACKUP_GATE_CHECK_ENABLED": "0",
        }
    )
    if extra_env:
        env.update(extra_env)
    staged = _stage(script, tmp_path, bin_dir)
    proc = subprocess.run(
        ["bash", str(staged), "--mode", "alert"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (
        f"alert run failed rc={proc.returncode} stderr={proc.stderr[-2000:]}"
    )
    logs = sorted((tmp_path / "sandbox" / "logs").glob("*.log"))
    log_text = logs[-1].read_text() if logs else proc.stdout
    post_texts: list[str] = []
    if posts.is_dir():
        for payload in sorted(posts.glob("*.json")):
            post_texts.append(json.loads(payload.read_text()).get("text", ""))
    return log_text, post_texts


def _alert_state(tmp_path: Path) -> str:
    """Per-key alert state, or '' when the reporter wrote none (OMN-16789)."""
    state = tmp_path / "sandbox" / "state" / "omninode-system-alert-keys.tsv"
    return state.read_text() if state.exists() else ""


def _set_http(tmp_path: Path, port: str, entry: tuple[int, str] | None) -> None:
    """Repoint one port in the live stub spec between ticks.

    ``entry=None`` removes the port, which the stub curl treats as a refused
    connection (empty body, non-zero exit) -- the same ``000`` the real reporter
    scores when a probe times out. That is what the measured 18085 flap was.
    """
    spec_path = tmp_path / "spec.json"
    spec = json.loads(spec_path.read_text())
    if entry is None:
        spec["http"].pop(str(port), None)
    else:
        spec["http"][str(port)] = list(entry)
    spec_path.write_text(json.dumps(spec))


class _AlertTicker:
    """Drive ``--mode alert`` repeatedly against one persistent state dir.

    The cadence logic under test is inherently multi-tick: confirmation,
    absence-hysteresis and re-notification cannot be observed in a single run.
    ``_run_alert`` renames the stub curl on every call, so it cannot simply be
    called twice; this installs the Slack-aware curl once and then re-runs the
    staged script, returning only the posts produced by THAT tick.
    """

    def __init__(
        self,
        script: Path,
        tmp_path: Path,
        bin_dir: Path,
        *,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.posts_dir = tmp_path / "slack-posts"
        self.extra_env = dict(extra_env or {})
        self._seen: set[Path] = set()

        real_curl = bin_dir / "curl-http"
        (bin_dir / "curl").rename(real_curl)
        _write(
            bin_dir / "curl",
            _SLACK_CURL.replace("__POSTS__", str(self.posts_dir)).replace(
                "__REALCURL__", str(real_curl)
            ),
            executable=True,
        )
        self.staged = _stage(script, tmp_path, bin_dir)

    def tick(self) -> tuple[str, list[str]]:
        env = dict(os.environ)
        env.update(
            {
                "OMNINODE_ALERT_ENV_FILE": str(self.tmp_path / "absent.env"),
                "OMNINODE_INFRA_REPO_ROOT": str(REPO_ROOT),
                "OMNINODE_RUNTIME_POLICY_ENV": str(RUNTIME_POLICY_ENV),
                "SLACK_BOT_TOKEN": "test-token",
                "SLACK_CHANNEL_ID": "C-TEST",
                "OMNINODE_CI_PROBE_ENABLED": "0",
                # OMN-18567: same reasoning one row up. The reporter also reads
                # the runner-tree converge verdict from collect(). These tests
                # are about disk/docker/endpoint classification and must not
                # acquire a dependency on a state file that a cron tick writes;
                # the collector's own rows are asserted in the OMN-18567 block
                # at the end of this file, which switches it back on.
                "OMNINODE_RUNNER_TREE_CHECK_ENABLED": "0",
                # OMN-18944: same reasoning again. The reporter also reads the
                # backup freshness gate's latest run from collect(). These tests
                # must not acquire a GitHub network dependency; that check's own
                # rows are asserted in
                # tests/scripts/test_postgres_backup_freshness_row_omn18944.py,
                # which drives it through its fetch seam against recorded payloads.
                "OMNINODE_BACKUP_GATE_CHECK_ENABLED": "0",
            }
        )
        env.update(self.extra_env)
        proc = subprocess.run(
            ["bash", str(self.staged), "--mode", "alert"],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 0, (
            f"alert run failed rc={proc.returncode} stderr={proc.stderr[-2000:]}"
        )
        logs = sorted((self.tmp_path / "sandbox" / "logs").glob("*.log"))
        log_text = logs[-1].read_text() if logs else proc.stdout

        fresh: list[str] = []
        if self.posts_dir.is_dir():
            for payload in sorted(self.posts_dir.glob("*.json")):
                if payload in self._seen:
                    continue
                self._seen.add(payload)
                fresh.append(json.loads(payload.read_text()).get("text", ""))
        return log_text, fresh


def test_alert_mode_pages_when_a_runtime_lane_is_down(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """A 503 dev lane must produce a Slack alert naming that lane.

    RED-before: `$issues` was selected with `$2=="CRITICAL"`, but endpoint rows
    carry their status in `$1`. With only endpoints failing, `$issues` was
    empty, the alert branch wrote "clean" to the state file, and nothing was
    ever posted -- while the digest text simultaneously listed the lane as
    CRITICAL. The reporter could see the dead runtime and still not page.

    OMN-16789 changed the cadence, not the outcome: a key must hold the same
    status for ``OMNINODE_ALERT_CONFIRM_TICKS`` before it may page, so the alert
    lands on the confirming tick rather than the first sighting. The assertion
    that matters -- a 503 dev lane produces a Slack post naming that lane -- is
    unchanged, and the first-tick silence is itself asserted below so this
    cannot pass against a script that has simply gone mute.
    """
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_outage_http(lane_ports),
        docker_state={"containers": [], "dangling": []},
    )
    ticker = _AlertTicker(FIXED_SCRIPT, tmp_path, bin_dir)

    log_text, posts = ticker.tick()
    assert not posts, (
        "alert paged on the first sighting -- CONFIRM_TICKS was not honoured, so "
        f"a single-tick blip can page.\nlog:\n{log_text}"
    )

    log_text, posts = ticker.tick()
    assert posts, (
        "no Slack post was attempted on the confirming tick while the dev runtime "
        f"lane was 503 (OMN-15525/OMN-16789).\nlog:\n{log_text}"
    )
    joined = "\n".join(posts)
    assert f"runtime-dev-{lane_ports['dev']}" in joined, (
        f"alert fired but never named the dead dev lane:\n{joined}"
    )
    assert re.search(r"Issues: \*[1-9]\d* critical\*", joined), joined

    state = _alert_state(tmp_path)
    assert f"runtime-dev-{lane_ports['dev']}" in state, (
        f"alert run did not record the failing lane in per-key state:\n{state}"
    )


def test_alert_mode_stays_quiet_on_a_healthy_fleet(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """Control for the test above: an all-green fleet must post nothing.

    Without this, `test_alert_mode_pages_when_a_runtime_lane_is_down` could pass
    against a script that posts unconditionally. Driven for more ticks than
    CONFIRM_TICKS so "quiet" means quiet, not merely un-confirmed yet.
    """
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_all_green_http(lane_ports),
        docker_state={"containers": [], "dangling": []},
    )
    ticker = _AlertTicker(FIXED_SCRIPT, tmp_path, bin_dir)
    for _ in range(3):
        log_text, posts = ticker.tick()
        assert not posts, (
            f"alert posted against a fully healthy fleet:\n{posts}\n{log_text}"
        )

    assert _alert_state(tmp_path).strip() == "", (
        f"healthy fleet left keys in alert state: {_alert_state(tmp_path)!r}"
    )


# --------------------------------------------------------------------------
# OMN-16789 -- the operator's actual complaint: the same alert, over and over.
#
# The de-duplication was not absent, it was DEFEATED. It hashed the whole issue
# SET, which is sound only against a stable input, and the input was not stable:
# `runtime-stability-test-18085` bounced CRITICAL(000)/OK(200) on nearly every
# tick, so the set alternated and the hash changed every time. Measured on .201
# from /data/maintenance/logs/ across 39 ticks (2026-08-27 09:30Z-18:30Z):
# 22 posted, 17 suppressed, every post traceable to that one key bouncing.
#
# `_MEASURED_18085_FLAP` below is that observed sequence, transcribed from the
# tick logs. It is the RED-before: against the set-hash revision it produces a
# post on nearly every element.
# --------------------------------------------------------------------------

# CRITICAL(True) / OK(False) for runtime-stability-test-18085, 2026-08-27,
# 13:00Z-18:30Z, read off the per-tick logs on .201.
_MEASURED_18085_FLAP = [
    True, False, False, True, False, True, False, True, False, False,
    False, False, False, False, False, True, False, False, False, False,
    True, False, True,
]  # fmt: skip


def test_flapping_key_does_not_repost_the_alert(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """The regression test for the reported spam.

    Replays the measured 18085 flap. The key is genuinely bad some ticks and
    genuinely fine others; what must NOT happen is a fresh alert on each bounce.

    RED-before (set-hash revision): every transition changes the set hash, so
    this posts on the order of a dozen times across the sequence -- and the
    shrink direction posts the FULL alert digest again rather than reading as a
    recovery, which is exactly what the operator screenshotted.
    """
    http = dict(_all_green_http(lane_ports))
    stab_port = lane_ports["stability-test"]
    bin_dir = _make_stub_bin(
        tmp_path, http=http, docker_state={"containers": [], "dangling": []}
    )
    ticker = _AlertTicker(FIXED_SCRIPT, tmp_path, bin_dir)

    total_posts: list[str] = []
    for critical in _MEASURED_18085_FLAP:
        if critical:
            _set_http(tmp_path, stab_port, None)  # unreachable -> code 000
        else:
            _set_http(tmp_path, stab_port, _all_green_http(lane_ports)[stab_port])
        _, posts = ticker.tick()
        total_posts.extend(posts)

    assert len(total_posts) <= 2, (
        f"{len(total_posts)} Slack posts across {len(_MEASURED_18085_FLAP)} ticks "
        "of a single flapping key -- the flap is re-arming the alert. Posts:\n"
        + "\n---\n".join(total_posts)
    )
    # And it must not have gone mute: the key really was critical, so the one
    # alert it is allowed must name it.
    assert total_posts, "flapping critical key produced no alert at all"
    assert f"runtime-stability-test-{stab_port}" in "\n".join(total_posts)


def test_standing_critical_is_renotified_on_the_long_interval(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """A permanent critical must not go permanently silent.

    Suppressing repeats is only correct if the standing condition is still
    re-surfaced on some cadence -- otherwise the fix for noise is a new
    false-green.

    The standing critical replayed here is the dev lane's 503 from
    `_outage_http`, not the `deploy-agent-8099` row this docstring used to
    cite: that probe was a phantom and was removed in the OMN-16789 follow-up
    (see `test_phantom_deploy_agent_endpoint_is_not_probed`). The test never
    actually depended on it -- worth stating, because "a standing critical
    exists" is this test's premise and it must come from a real one.
    """
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_outage_http(lane_ports),
        docker_state={"containers": [], "dangling": []},
    )
    # Re-notify immediately so the test does not sleep 6 hours; the point under
    # test is that the interval is honoured and env-driven, not its default.
    ticker = _AlertTicker(
        FIXED_SCRIPT,
        tmp_path,
        bin_dir,
        extra_env={"OMNINODE_ALERT_RENOTIFY_SECONDS": "0"},
    )
    ticker.tick()  # sighting
    _, first = ticker.tick()  # confirmed -> NEW
    assert first, "standing critical never produced its first alert"

    ticker.extra_env = {"OMNINODE_ALERT_RENOTIFY_SECONDS": "1"}
    time.sleep(1.1)
    _, second = ticker.tick()
    assert second, (
        "a still-standing critical was never re-notified -- suppression became "
        "permanent silence"
    )


def test_recovery_names_the_key_and_does_not_repost_the_digest(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """A key clearing must read as a recovery, not as the same alert again.

    RED-before: with any other issue still standing, `$issues` was non-empty, so
    the clearing tick took the ALERT branch and re-posted the whole digest minus
    one line. That is the message the operator saw repeatedly.
    """
    dev_port = lane_ports["dev"]
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_outage_http(lane_ports),
        docker_state={"containers": [], "dangling": []},
    )
    ticker = _AlertTicker(
        FIXED_SCRIPT, tmp_path, bin_dir, extra_env={"OMNINODE_ALERT_CLEAR_TICKS": "2"}
    )
    ticker.tick()
    _, alert = ticker.tick()
    assert alert, "no initial alert to recover from"

    _set_http(tmp_path, dev_port, _all_green_http(lane_ports)[dev_port])
    _, none_yet = ticker.tick()
    assert not none_yet, "recovery fired before CLEAR_TICKS absences elapsed"
    _, recovery = ticker.tick()

    assert recovery, "key cleared but no recovery was ever posted"
    text = "\n".join(recovery)
    assert "alert resolved" in text.lower(), f"not a recovery message:\n{text}"
    assert f"runtime-dev-{dev_port}" in text, (
        f"recovery did not name the key that recovered:\n{text}"
    )
    assert "*Runtime endpoints*" not in text, (
        "the recovery re-posted the full alert digest -- this is the exact "
        f"OMN-16789 behaviour under test:\n{text}"
    )


def test_alert_cadence_and_probe_timeout_carry_no_hardcoded_literals() -> None:
    """AC3/AC4: the decision path must be env-driven, not literal.

    The 4-second probe ceiling was the root of the flap (a warm .201 lane
    answers in ~3.2s under load), and it was unreachable from config. A tunable
    that only exists as a literal cannot be tuned when the host gets slower.
    """
    body = FIXED_SCRIPT.read_text()
    for var, env in (
        ("PROBE_TIMEOUT_SECONDS", "OMNINODE_ALERT_PROBE_TIMEOUT_SECONDS"),
        ("CONFIRM_TICKS", "OMNINODE_ALERT_CONFIRM_TICKS"),
        ("CLEAR_TICKS", "OMNINODE_ALERT_CLEAR_TICKS"),
        ("RENOTIFY_SECONDS", "OMNINODE_ALERT_RENOTIFY_SECONDS"),
    ):
        assert re.search(rf"^{var}=\$\{{{env}:-", body, re.MULTILINE), (
            f"{var} is not overridable via {env}"
        )
    # Comment lines are exempt: the header documents the old `--max-time 4` as
    # the root cause, and that prose is why the next reader understands the
    # knob. Only executable lines are asserted on.
    code = [ln for ln in body.splitlines() if not ln.lstrip().startswith("#")]
    offenders = [ln for ln in code if re.search(r"--max-time\s+4\b", ln)]
    assert not offenders, (
        f"a hardcoded 4s probe timeout survives in executable code: {offenders}"
    )


def test_endpoint_failures_are_counted_in_the_header(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """The header count must agree with the *Active issues* list.

    The two were computed by different awk programs over different columns, so
    the digest could say `0 critical` directly above three CRITICAL lanes --
    observed verbatim on .201 against the merged OMN-15509 revision.
    """
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_outage_http(lane_ports),
        docker_state={"containers": [], "dangling": []},
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)

    header = re.search(r"Issues: \*(\d+) critical\*, \*(\d+) warning\*", report)
    assert header, report
    listed_critical = len(
        re.findall(r"^- CRITICAL ", report.split("*Active issues*", 1)[1], re.MULTILINE)
    )
    assert int(header.group(1)) == listed_critical, (
        f"header claims {header.group(1)} critical but {listed_critical} are "
        f"listed under *Active issues*:\n{report}"
    )
    assert listed_critical > 0, report


# --------------------------------------------------------------------------
# OMN-18435 -- a lane inside its container's declared start_period is BOOTING,
# not down.
#
# Measured on .201 over 242 ticks (2026-09-14T00:00Z -> 2026-09-16T11:45Z):
# `runtime-dev-8085` was CRITICAL on 20 ticks, EVERY one a single isolated tick
# with the next tick back at 200. Eleven of the fourteen 503s carried
# `is_running=false` -- a runtime serving HTTP whose kernel had not started, i.e.
# a boot. `RestartCount` was 0 throughout, so nothing crashed. The lane is
# recreated by the delivery chain dozens of times a day and its image declares
# `StartPeriod=1800s`, so Docker itself never calls it unhealthy during these
# boots -- only this reporter did.
#
# The script already honours a container's declared start_period on the
# CONTAINER path (`starting_past_start_period`). These tests pin the same
# concept on the ENDPOINT path, and pin fail-closed everywhere it cannot be
# established: a monitor that cannot tell is still not allowed to say green.
# --------------------------------------------------------------------------


def _iso_ago(seconds: int) -> str:
    """A UTC `StartedAt` that many seconds in the past, in Docker's own shape."""
    return time.strftime(
        "%Y-%m-%dT%H:%M:%S.000000000Z", time.gmtime(time.time() - seconds)
    )


def _lane_docker(
    dev_port: str,
    *,
    age_seconds: int,
    start_period_seconds: int,
    publish_port: bool = True,
    started_at: str | None = None,
) -> dict[str, Any]:
    """Docker state whose dev-lane runtime publishes the dev lane's main port.

    Mirrors `.201`: `docker ps --format '{{.Names}}\t{{.Ports}}'` there renders
    `omninode-runtime  0.0.0.0:8085->8085/tcp, [::]:8085->8085/tcp`, which is the
    only fact tying a probed port to a container. Nothing here hardcodes the
    container NAME into the script under test -- the port is the join key, and
    it is the same port the probe itself already resolved from policy.
    """
    ports = (
        f"0.0.0.0:{dev_port}->8085/tcp, [::]:{dev_port}->8085/tcp"
        if publish_port
        else ""
    )
    return {
        "containers": [
            {
                "name": "omninode-runtime",
                "status": "Up 1 minute (health: starting)",
                "started_at": (
                    started_at if started_at is not None else _iso_ago(age_seconds)
                ),
                "start_period_ns": start_period_seconds * 10**9,
                "ports": ports,
            },
            {
                "name": "omnibase-infra-postgres",
                "status": "Up 40 minutes (healthy)",
                "started_at": _iso_ago(2400),
                "ports": "0.0.0.0:5432->5432/tcp",
            },
        ],
        "dangling": [],
    }


def _dev_endpoint_status(report: str, dev_port: str) -> str:
    """The dev lane's status as the digest renders it under *Runtime endpoints*.

    `_run` returns the rendered digest, not the raw snapshot, so the status is
    read from the line the operator actually sees -- which also pins that a
    non-CRITICAL lane is still RENDERED rather than silently dropped.
    """
    section = report.split("*Runtime endpoints*", 1)
    if len(section) < 2:
        return ""
    body = section[1].split("*CI required contexts*", 1)[0]
    match = re.search(
        rf"^- `runtime-dev-{re.escape(dev_port)}`: HTTP \S+ \((\w+)\)$",
        body,
        re.MULTILINE,
    )
    return match.group(1) if match else ""


def test_runtime_lane_inside_its_start_period_is_not_critical(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC1 -- a 503 from a lane 60s into a 1800s start_period is a boot.

    This is the measured case: 11 of the 20 critical ticks carried
    `is_running=false` while the container had just been recreated.
    """
    dev_port = lane_ports["dev"]
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_outage_http(lane_ports),
        docker_state=_lane_docker(dev_port, age_seconds=60, start_period_seconds=1800),
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)

    status = _dev_endpoint_status(report, dev_port)
    assert status, f"the dev lane vanished from *Runtime endpoints* entirely:\n{report}"
    assert status != "CRITICAL", (
        f"a lane 60s into a 1800s start_period was scored CRITICAL:\n{report}"
    )
    assert status == "STARTING", (
        f"expected a distinct non-paging status for a booting lane, got {status}:\n{report}"
    )
    # A booting lane must not become an ACTIVE ISSUE, and the header count must
    # agree -- the OMN-15525 failure was a header that disagreed with the rows.
    issues = report.split("*Active issues*", 1)[1]
    assert f"runtime-dev-{dev_port}" not in issues, (
        f"a booting lane was listed under *Active issues*:\n{report}"
    )


def test_runtime_lane_inside_its_start_period_does_not_page(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC1 -- and it never reaches the alert key set, so it cannot page."""
    dev_port = lane_ports["dev"]
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_outage_http(lane_ports),
        docker_state=_lane_docker(dev_port, age_seconds=60, start_period_seconds=1800),
    )
    _log, posts = _run_alert(FIXED_SCRIPT, tmp_path, bin_dir)

    assert f"runtime-dev-{dev_port}" not in _alert_state(tmp_path), (
        f"a booting lane entered the alert state machine:\n{_alert_state(tmp_path)}"
    )
    assert not any(f"runtime-dev-{dev_port}" in text for text in posts), (
        f"a booting lane paged Slack:\n{posts}"
    )


def test_runtime_lane_past_its_start_period_is_still_critical(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC2 -- the grace is bounded by the container's OWN declared budget."""
    dev_port = lane_ports["dev"]
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_outage_http(lane_ports),
        docker_state=_lane_docker(
            dev_port, age_seconds=7200, start_period_seconds=1800
        ),
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)

    assert _dev_endpoint_status(report, dev_port) == "CRITICAL", (
        f"a lane 2h into a 1800s start_period must still be CRITICAL:\n{report}"
    )


def test_runtime_lane_with_no_container_on_the_port_fails_closed(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC2 -- no container publishes the port, so nothing proves a boot."""
    dev_port = lane_ports["dev"]
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_outage_http(lane_ports),
        docker_state=_lane_docker(
            dev_port, age_seconds=60, start_period_seconds=1800, publish_port=False
        ),
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)

    assert _dev_endpoint_status(report, dev_port) == "CRITICAL", (
        f"an unresolvable container must fail closed to CRITICAL:\n{report}"
    )


def test_runtime_lane_with_unparseable_start_time_fails_closed(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC2 -- a container that cannot be AGED cannot be proven inside its grace.

    Same rule the container path already applies (`age-unknown`): losing the
    ability to measure must never be the quiet outcome.
    """
    dev_port = lane_ports["dev"]
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_outage_http(lane_ports),
        docker_state=_lane_docker(
            dev_port,
            age_seconds=60,
            start_period_seconds=1800,
            started_at="not-a-timestamp",
        ),
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)

    assert _dev_endpoint_status(report, dev_port) == "CRITICAL", (
        f"a container with an unparseable StartedAt must fail closed:\n{report}"
    )


def test_healthy_lane_inside_its_start_period_still_reads_ok(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """Positive control -- the new branch only ever downgrades a CRITICAL.

    Without this, a bug that stamped STARTING on everything inside the grace
    would pass every test above while destroying the OK signal.
    """
    dev_port = lane_ports["dev"]
    http = _outage_http(lane_ports)
    http[dev_port] = (200, HEALTHY_BODY)
    bin_dir = _make_stub_bin(
        tmp_path,
        http=http,
        docker_state=_lane_docker(dev_port, age_seconds=60, start_period_seconds=1800),
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)

    assert _dev_endpoint_status(report, dev_port) == "OK", (
        f"a healthy lane inside its start_period must still read OK:\n{report}"
    )


# The unhealthy dimension is placed deliberately past the display-excerpt
# boundary. On .201 the real body is ~4.3 KB and `BODY_EXCERPT_BYTES` is 180, so
# every recorded 503 row stopped before `event_bus_healthy` -- which is why the
# three genuinely-unhealthy ticks in the measured window cannot be diagnosed
# from the logs at all. The monitor recorded a verdict with no reason.
_DEEP_UNHEALTHY_BODY = json.dumps(
    {
        "status": "unhealthy",
        "version": "0.38.29",
        "details": {
            "healthy": False,
            "degraded": False,
            "startup_in_progress": False,
            "is_running": True,
            "is_draining": False,
            "pending_message_count": 0,
            "max_concurrent_handlers": 64,
            "handler_pool_size": 8,
            "in_flight_tasks": 0,
            "batch_response_enabled": True,
            "batch_response_pending": 0,
            "event_bus_healthy": False,
            "no_handlers_registered": False,
            "registered_handlers": 41,
        },
    }
)
assert _DEEP_UNHEALTHY_BODY.index("event_bus_healthy") > 180, (
    "the failing dimension must sit past the display-excerpt boundary or this "
    "test cannot observe the truncation it exists to pin"
)


def test_unhealthy_runtime_detail_names_the_failing_dimension(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """AC3 -- record WHY, not just that."""
    dev_port = lane_ports["dev"]
    http = _outage_http(lane_ports)
    http[dev_port] = (503, _DEEP_UNHEALTHY_BODY)
    bin_dir = _make_stub_bin(
        tmp_path,
        http=http,
        docker_state=_lane_docker(
            dev_port, age_seconds=7200, start_period_seconds=1800
        ),
    )
    report = _run(FIXED_SCRIPT, tmp_path, bin_dir)

    assert _dev_endpoint_status(report, dev_port) == "CRITICAL", report
    issues = report.split("*Active issues*", 1)[1]
    assert "event_bus_healthy=false" in issues, (
        "the recorded row must name the failing health dimension; the 180-byte "
        f"excerpt alone stops before it:\n{issues}"
    )


# --------------------------------------------------------------------------
# OMN-18567 -- the runner-tree converge verdict reaches a human.
#
# `omninode-runner-tree-converge.sh` runs hourly at :49 over the deploy
# runner's private clone tree -- the build source the dev-lane refresh, the
# stability-lane refresh and the release-train tag cut all read -- and writes
# one verdict line. A verdict nothing reads is a log line (CLAUDE.md rule 5),
# so the reporter collects it here rather than in a second alerter.
#
# The load-bearing distinction these tests pin: the tick REFUSES while a deploy
# job is using the tree and exits 0 when it does, because a refusal is the guard
# working. So a single REFUSED must NOT page. What pages is the tree going
# unconverged for a long time, which is detected by ageing `last_success` --
# and that catches "permanently busy", "unit stopped being scheduled" and
# "nobody ever installed it" alike, three conditions that look identical from
# the outside.
# --------------------------------------------------------------------------
def _runner_tree_rows(
    tmp_path: Path, lane_ports: dict[str, str], status_line: str | None
) -> tuple[list[str], list[str]]:
    """(digest rows, rows that reached *Active issues*) for the runner-tree key."""
    bin_dir = _make_stub_bin(
        tmp_path,
        http=_all_green_http(lane_ports),
        docker_state={"containers": [], "dangling": []},
    )
    status_file = tmp_path / "runner-tree-converge.status"
    if status_line is not None:
        status_file.write_text(status_line + "\n", encoding="utf-8")
    report = _run(
        FIXED_SCRIPT,
        tmp_path,
        bin_dir,
        extra_env={
            "OMNINODE_RUNNER_TREE_CHECK_ENABLED": "1",
            "OMNINODE_RUNNER_TREE_STATE_FILE": str(status_file),
        },
    )
    # OMN-18944 inserted the backup freshness section after this one, so the
    # runner-tree section now ends at that header rather than at *Active
    # issues*. Slicing to the next header rather than to the end of the digest
    # is what keeps this assertion about the runner-tree rows only.
    section = report.split("*Runner clone tree*", 1)[1].split(
        "*Production database backup*", 1
    )[0]
    issues = report.split("*Active issues*", 1)[1]
    rows = [
        line.strip() for line in section.splitlines() if line.strip().startswith("- ")
    ]
    paged = [
        line.strip()
        for line in issues.splitlines()
        if line.strip().startswith("- ") and "`converge`" in line
    ]
    return rows, paged


def _verdict_line(verdict: str, last_success: str, detail: str = "r:aaa->bbb") -> str:
    return (
        f"runner-tree-converge|{verdict}|ts=2026-09-17T12:00:00Z"
        f"|tree=/data/omninode/runner_omni_home|clones=6|in_sync=6|converged=0"
        f"|failed=0|skipped=0|reason=none|last_success={last_success}|detail={detail}"
    )


def _hours_ago(hours: int) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_a_missing_verdict_file_is_a_warning_not_silence(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """ "Nobody installed it" must look different from "it ran and was fine".

    An artifact merged into the repo but never installed on the host, with
    nothing alarming, is the OMN-15525 condition -- the exact failure this
    maintenance family exists to make visible.
    """
    rows, _paged = _runner_tree_rows(tmp_path, lane_ports, None)
    assert len(rows) == 1, rows
    assert "(WARNING)" in rows[0]
    assert "never run" in rows[0]


def test_a_failed_converge_is_critical(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """The one outcome worth waking someone for: the tree is knowingly wrong."""
    rows, _paged = _runner_tree_rows(
        tmp_path,
        lane_ports,
        _verdict_line("FAILED", _hours_ago(0), detail="omnibase_infra:aaa->aaa"),
    )
    assert "(CRITICAL)" in rows[0]
    assert "omnibase_infra" in rows[0]


def test_a_fresh_converge_is_ok(tmp_path: Path, lane_ports: dict[str, str]) -> None:
    rows, _paged = _runner_tree_rows(
        tmp_path, lane_ports, _verdict_line("CONVERGED", _hours_ago(0))
    )
    assert "(OK)" in rows[0]


def test_a_single_refusal_with_a_recent_success_does_not_page(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """A busy runner is the guard working, not a fault.

    Paging on every refusal is how a channel stops being read, which is this
    reporter's own stated failure mode.
    """
    rows, paged = _runner_tree_rows(
        tmp_path, lane_ports, _verdict_line("REFUSED", _hours_ago(1))
    )
    assert "(OK)" in rows[0]
    assert not paged, f"a single refusal reached *Active issues*: {paged}"


def test_a_tree_unconverged_for_too_long_warns_even_while_refusing(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """The case the verdict alone cannot express.

    A permanently busy runner refuses forever and exits 0 forever. Reading only
    the latest verdict, that is indistinguishable from a permanently converged
    tree. Ageing `last_success` is what separates them.
    """
    rows, _paged = _runner_tree_rows(
        tmp_path, lane_ports, _verdict_line("REFUSED", _hours_ago(30))
    )
    assert "(WARNING)" in rows[0]
    assert "30h ago" in rows[0]


def test_a_tick_that_never_succeeded_warns(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    rows, _paged = _runner_tree_rows(
        tmp_path, lane_ports, _verdict_line("REFUSED", "never")
    )
    assert "(WARNING)" in rows[0]
    assert "never been converged" in rows[0]


def test_an_unparseable_verdict_is_a_warning_not_a_green(
    tmp_path: Path, lane_ports: dict[str, str]
) -> None:
    """ "Could not look" must never render as "nothing wrong"."""
    rows, _paged = _runner_tree_rows(
        tmp_path, lane_ports, "garbage that is not a verdict"
    )
    assert "(WARNING)" in rows[0]
    assert "unparseable" in rows[0]
