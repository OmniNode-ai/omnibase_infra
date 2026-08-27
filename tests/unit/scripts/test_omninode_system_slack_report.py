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


def _script_lane_specs(script: Path) -> dict[str, str]:
    """Parse the reporter's own RUNTIME_LANE_SPECS array into {lane: policy_key}."""
    match = RUNTIME_LANE_SPECS_RE.search(script.read_text())
    assert match, f"RUNTIME_LANE_SPECS array not found in {script}"
    specs: dict[str, str] = {}
    for row in re.findall(r'"([^"]+)"', match.group("body")):
        lane, _, key = row.partition("|")
        specs[lane] = key
    assert specs, f"RUNTIME_LANE_SPECS in {script} parsed to an empty map"
    return specs


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
def ps_all():
    return [(c["name"], c["status"]) for c in spec["containers"]]
def running():
    return [c for c in spec["containers"] if c["status"].startswith("Up")]
if a[0]=="ps" and "-a" in a:
    for n,s in ps_all(): print(f"{{n}}\\t{{s}}")
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
    print(c["started_at"] if "StartedAt" in fmt else c.get("start_period_ns",0))
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
            "8099": (200, json.dumps({"state": "idle"})),
            "3003": (200, "<html>ok</html>"),
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
    return {lane: _policy_port(key) for lane, key in LANE_PORT_KEYS.items()}


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
    assert f"runtime-{lane_ports['prod']}`: HTTP 200 (OK)" in endpoints
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
    assert f"runtime-prod-{lane_ports['prod']}`: HTTP 200 (OK)" in report
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
            "8099": (200, '{"state":"idle"}'),
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

    unprobed = sorted(set(LANE_PORT_KEYS) - set(script_lanes))
    assert not unprobed, (
        f"lane(s) {unprobed} declare a *_RUNTIME_MAIN_PORT in "
        f"{RUNTIME_POLICY_ENV.name} but are absent from RUNTIME_LANE_SPECS in "
        f"{FIXED_SCRIPT.name} -- a live runtime lane whose death pages nobody"
    )
    phantom = sorted(set(script_lanes) - set(LANE_PORT_KEYS))
    assert not phantom, (
        f"lane(s) {phantom} appear in RUNTIME_LANE_SPECS but declare no "
        f"*_RUNTIME_MAIN_PORT in {RUNTIME_POLICY_ENV.name} -- the port can "
        f"never resolve, so the reporter alarms forever on a phantom lane"
    )
    assert script_lanes == LANE_PORT_KEYS, (
        f"lane -> policy-key mapping disagrees between the reporter and the "
        f"policy: script={script_lanes} policy={LANE_PORT_KEYS}"
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
    specs = re.search(r"RUNTIME_LANE_SPECS=\((.*?)\n\)", source, re.DOTALL)
    assert specs, "RUNTIME_LANE_SPECS table not found"
    for raw in specs.group(1).strip().splitlines():
        entry = raw.strip().strip('"')
        if not entry:
            continue
        fields = entry.split("|")
        assert len(fields) == 2, (
            f"lane spec {entry!r} carries more than lane|key -- a third field is "
            "the hardcoded fallback port OMN-15525 removed"
        )
        assert not re.fullmatch(r"\d+", fields[1]), entry


def test_cron_unit_points_at_the_versioned_script_name() -> None:
    unit = CRON_UNIT.read_text()
    assert "omninode-system-slack-report.sh" in unit
    assert "--mode alert" in unit and "--mode digest" in unit


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
            "8099": (200, '{"state":"idle"}'),
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
            "8099": (200, '{"state":"idle"}'),
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
            "8099": (200, '{"state":"idle"}'),
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
            "8099": (200, '{"state":"idle"}'),
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

    for lane in LANE_PORT_KEYS:
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
    partial.write_text(
        "".join(
            f"{key}_RENAMED={lane_ports[lane]}\n"
            if lane == "dev"
            else f"{key}={lane_ports[lane]}\n"
            for lane, key in LANE_PORT_KEYS.items()
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

    `deploy-agent-8099` was CRITICAL on all 39 measured ticks. Suppressing
    repeats is only correct if the standing condition is still re-surfaced on
    some cadence -- otherwise the fix for noise is a new false-green.
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
