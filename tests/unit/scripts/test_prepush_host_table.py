# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Guards for lab-wide pre-push distribution (OMN-16991).

Three things are pinned here, because all three were structural defects rather
than bugs in a computation:

1. **The host table is the identity authority, read from the COMMITTED tree.**
   The guard used to test two hard-coded hostnames -- a literal ``||`` that was
   the entire reason ``.101``/``.105`` could not be used. The full table
   contents are asserted, so adding or promoting a host requires a reviewed
   commit *and* a deliberate edit here.

2. **Placement reads SLOT state before load.** Measured 2026-08-30: ``.201``
   showed the fittest load ratio in the lab (14.08/32 = 0.44x) while running
   three concurrent pre-push suites behind a 10-deep queue. A load-only picker
   routes a fourth run onto the most jammed host in the fleet.

3. **Nothing here may make the gate accept less work.** The precedence tests
   pin the GitHub-hosted sha-pinned run ahead of the lab leg, and pin that a
   remote RED refuses instead of falling through to the override grant.

The bash helpers are extract-and-executed (the pattern already used for this
hook's other pure shell functions) so the assertions run THE code that ships,
never a Python re-implementation that could pass while the shipped picker is
broken.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HOOK = REPO_ROOT / "scripts" / "hooks" / "prepush_smart_tests.sh"
LIB = REPO_ROOT / "scripts" / "hooks" / "prepush_dispatch.sh"
TABLE = REPO_ROOT / "scripts" / "hooks" / "prepush_hosts.tsv"

pytestmark = pytest.mark.unit


# =============================================================================
# The table itself
# =============================================================================


def _rows() -> list[list[str]]:
    rows = []
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]
        if not line.strip():
            continue
        rows.append(line.split("\t"))
    return rows


def test_table_exists_and_every_row_has_the_full_column_set() -> None:
    assert TABLE.is_file(), f"expected the host table at {TABLE}"
    rows = _rows()
    assert rows, "expected at least one data row"
    for row in rows:
        assert len(row) == 12, (
            f"row {row[0] if row else row!r} has {len(row)} columns, expected 12 "
            "(label role hostname ssh_target cores uv_abs_path uv_min_version "
            "workroot slot_mode repos_denied mode note)"
        )


def test_table_contents_are_pinned() -> None:
    """The exact designated set, asserted.

    This is the point of the file: the table decides which machines may
    authorize a heavy gate run, so a row addition or a `mode` promotion must be
    a reviewed, deliberate change and not a quiet edit.
    """
    got = {r[0]: (r[1], r[2], r[10]) for r in _rows()}
    assert got == {
        "h200": ("capacity", "stickybeatz-studio", "authorizing"),
        "h201": ("capacity", "omninode-pc", "authorizing"),
        "h201c": ("identity", "gate-runner-201", "authorizing"),
        "h101": ("capacity", "stickybeatz.local", "disabled"),
        "h105": ("capacity", "omnibook", "shadow"),
    }


def test_201_host_is_designated_by_its_real_hostname() -> None:
    """`.201`'s real `hostname -s` is `omninode-pc`; `gate-runner-201` is only
    the CONTAINER's. Before OMN-16991 only the container name was designated,
    so every push on the host itself needed an env override that the pytest
    child's env scrub then stripped."""
    hosts = {r[0]: r[2] for r in _rows()}
    assert hosts["h201"] == "omninode-pc"
    assert hosts["h201c"] == "gate-runner-201"


def test_201_is_denied_for_omnibase_infra_until_omn16989_closes() -> None:
    """The gate-runner fails 15 host-coupled `omnibase_infra` tests, so routing
    an infra push there produces a red that is not the diff's fault."""
    denied = {r[0]: r[9] for r in _rows()}
    assert "omnibase_infra" in denied["h201"].split(",")


def test_net_new_hosts_start_in_shadow_and_never_authorize() -> None:
    modes = {r[0]: r[10] for r in _rows()}
    assert modes["h105"] == "shadow"
    assert modes["h101"] == "disabled"


def test_every_capacity_row_carries_an_absolute_uv_path_and_a_floor() -> None:
    """uv is on no host's non-interactive PATH, and the live fleet spread is
    0.8.3 -> 0.11.32 against a lockfile at revision 3. Presence is not enough;
    the version floor is what makes a stale host skip rather than fail weirdly
    mid-`uv sync`."""
    for row in _rows():
        if row[1] != "capacity":
            continue
        assert row[5].startswith("/"), (
            f"{row[0]}: uv path must be absolute, got {row[5]!r}"
        )
        assert row[6][0].isdigit(), (
            f"{row[0]}: expected a uv_min_version, got {row[6]!r}"
        )


def test_101_workroot_avoids_the_tcc_protected_tree() -> None:
    """`ssh jonah@.101 'ls ~/Code'` returns `Operation not permitted`, so the
    workroot must live outside it -- the bundle design never needs `~/Code` on
    a remote host, which is what removes the out-of-band GUI grant step."""
    workroots = {r[0]: r[7] for r in _rows()}
    assert not workroots["h101"].startswith("/Users/jonah/Code")
    assert workroots["h101"] == "/Users/Shared/onex-prepush"


# =============================================================================
# Extract-and-execute harness
# =============================================================================


def _driver(repo_root: Path, body: str) -> str:
    """Run BODY with the real library sourced and the hook's own dependencies
    stubbed, against a throwaway git repo whose HEAD carries the real table."""
    script = f"""
set -uo pipefail
REPO_ROOT={repo_root}
PREPUSH_LOAD_THRESHOLD=1.0
log() {{ printf '[t] %s\\n' "$1" >&2; }}
die() {{ printf 'DIE: %s\\n' "$1" >&2; exit 1; }}
_prepush_timeout_cmd() {{ printf ''; }}
host_load_ratio() {{ return 1; }}
. {LIB}
{body}
"""
    completed = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={
            **os.environ,
            "PREPUSH_LOAD_OVERRIDE_MAP": "",
            "PREPUSH_SLOT_OVERRIDE_MAP": "",
        },
    )
    return completed.stdout


@pytest.fixture
def table_repo(tmp_path: Path) -> Path:
    """A throwaway repo whose HEAD carries the real table, so the tests
    exercise the real `git show HEAD:` read path rather than a stub."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "hooks").mkdir(parents=True)
    (repo / "scripts" / "hooks" / "prepush_hosts.tsv").write_text(
        TABLE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", "."], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "table"],
        cwd=repo,
        check=True,
    )
    return repo


# =============================================================================
# Identity
# =============================================================================


def test_identity_accepts_the_real_201_hostname(table_repo: Path) -> None:
    out = _driver(table_repo, "prepush_identity_label omninode-pc || echo NONE")
    assert out.strip() == "h201"


def test_identity_accepts_the_201_container_hostname(table_repo: Path) -> None:
    out = _driver(table_repo, "prepush_identity_label gate-runner-201 || echo NONE")
    assert out.strip() == "h201c"


def test_a_shadow_host_is_not_a_designated_identity(table_repo: Path) -> None:
    """A shadow host is a placement target whose verdict may not satisfy the
    escalation, so it must not confer identity either -- otherwise the identity
    guard would start PASSING on a host still in shadow, inverting the guard."""
    out = _driver(table_repo, "prepush_identity_label omnibook || echo NONE")
    assert out.strip() == "NONE"


def test_a_disabled_host_is_not_a_designated_identity(table_repo: Path) -> None:
    out = _driver(table_repo, "prepush_identity_label stickybeatz.local || echo NONE")
    assert out.strip() == "NONE"


def test_an_override_replaces_its_row_rather_than_adding_a_name(
    table_repo: Path,
) -> None:
    """OMN-15059's guard is proven by forcing a nonsense `PREPUSH_200_HOSTNAME`
    and asserting refusal. That only holds while the override REPLACES the .200
    row: an override that merely appended a name could no longer de-designate
    this machine, silently inverting the guard."""
    out = _driver(
        table_repo,
        "PREPUSH_200_HOSTNAME=nope prepush_identity_label stickybeatz-studio || echo NONE",
    )
    assert out.strip() == "NONE"


def test_the_per_row_override_can_de_designate_any_row(table_repo: Path) -> None:
    out = _driver(
        table_repo,
        "PREPUSH_HOST_OVERRIDE_H201=nope prepush_identity_label omninode-pc || echo NONE",
    )
    assert out.strip() == "NONE"


def test_an_uncommitted_table_edit_cannot_designate_a_host(table_repo: Path) -> None:
    """The table is read from HEAD and the working copy must agree. Otherwise a
    one-line uncommitted edit naming your laptop would self-authorize a heavy
    gate run with no review and no receipt -- the forgeable-artifact surface
    OMN-16688 deliberately avoided."""
    tsv = table_repo / "scripts" / "hooks" / "prepush_hosts.tsv"
    tsv.write_text(
        tsv.read_text(encoding="utf-8")
        + "hevil\tcapacity\tmy-laptop\t-\t8\t/bin/uv\t0.1.0\t/tmp/w\tlockdir\t-\tauthorizing\tforged\n",
        encoding="utf-8",
    )
    out = _driver(table_repo, "prepush_identity_label my-laptop || echo NONE")
    assert out.strip() == "NONE"


# =============================================================================
# The picker
# =============================================================================

_ALL_FREE = "h200=free,h201=free,h101=free,h105=free"


def _pick(
    repo: Path, *, load: str, slot: str, uv: str, repo_name: str = "omnibase_core"
) -> str:
    body = (
        f'export PREPUSH_LOAD_OVERRIDE_MAP="{load}"\n'
        f'export PREPUSH_SLOT_OVERRIDE_MAP="{slot}"\n'
        f'export PREPUSH_UV_OVERRIDE_MAP="{uv}"\n'
        f"if pick_capacity_host stickybeatz-studio {repo_name}; then\n"
        '  echo "PICK=$PREPUSH_PICK_LABEL"\n'
        "else\n"
        '  echo "PICK=none"\n'
        "fi\n"
        'echo "PROBE=$PREPUSH_PROBE_LOG"\n'
    )
    return _driver(repo, body)


_GOOD_UV = "h200=0.11.32,h201=0.11.5,h101=0.8.3,h105=0.11.8"


def test_picker_chooses_the_least_loaded_fit_host(table_repo: Path) -> None:
    out = _pick(
        table_repo,
        load="h200=0.90,h201=0.44,h105=0.21",
        slot=_ALL_FREE,
        uv=_GOOD_UV,
    )
    assert "PICK=h105" in out


def test_a_busy_host_is_unfit_even_when_it_is_the_least_loaded(
    table_repo: Path,
) -> None:
    """The measured case, not a hypothetical: `.201` read 0.44x -- the fittest
    ratio in the lab -- while running three concurrent pre-push suites behind a
    10-deep queue. load1 is a CPU-time proxy; the scarce resource is an
    exclusive heavy-suite slot."""
    out = _pick(
        table_repo,
        load="h200=0.90,h201=0.10,h105=0.80",
        slot="h200=free,h201=busy,h105=free",
        uv=_GOOD_UV,
    )
    assert "PICK=h105" in out, out
    assert "h201=busy" in out


def test_an_unreachable_host_is_skipped_never_assumed_free(
    table_repo: Path,
) -> None:
    """Silence is not headroom. A host we cannot read is skipped exactly like
    one we measured as over capacity -- the fail-closed posture the load probe
    already had."""
    out = _pick(
        table_repo,
        load="h200=0.90,h105=0.21",
        slot=_ALL_FREE,
        uv=_GOOD_UV,
    )
    assert "PICK=h105" in out
    assert "h201=unreachable" in out


def test_a_host_whose_slot_state_is_unknown_is_skipped(table_repo: Path) -> None:
    out = _pick(
        table_repo,
        load="h200=0.90,h201=0.10,h105=0.21",
        slot="h200=free,h201=unknown,h105=free",
        uv=_GOOD_UV,
    )
    assert "h201=slot-unknown" in out
    assert "PICK=h105" in out


def test_a_host_below_the_uv_floor_is_skipped(table_repo: Path) -> None:
    out = _pick(
        table_repo,
        load="h200=2.09,h201=2.0,h105=0.21",
        slot=_ALL_FREE,
        uv="h200=0.11.32,h201=0.11.5,h105=0.8.3",
    )
    assert "PICK=none" in out
    assert "h105=uv-unfit(0.8.3<0.11.0)" in out


def test_a_repo_denied_host_is_never_chosen(table_repo: Path) -> None:
    out = _pick(
        table_repo,
        load="h200=2.09,h201=0.10,h105=0.21",
        slot=_ALL_FREE,
        uv=_GOOD_UV,
        repo_name="omnibase_infra",
    )
    assert "h201=repo-denied" in out
    assert "PICK=h105" in out


def test_a_disabled_host_is_never_probed(table_repo: Path) -> None:
    out = _pick(table_repo, load="h101=0.01", slot=_ALL_FREE, uv="h101=9.9.9")
    assert "h101=disabled" in out
    assert "PICK=none" in out


def test_picker_returns_no_host_when_nothing_is_fit(table_repo: Path) -> None:
    """The fallback path. When no host is fit the picker must fail rather than
    return a least-bad guess -- the caller then falls through to the existing
    precedence (GitHub-hosted verify -> grant -> die), which is unchanged."""
    out = _pick(
        table_repo,
        load="h200=2.09,h201=3.10,h105=1.90",
        slot=_ALL_FREE,
        uv=_GOOD_UV,
    )
    assert "PICK=none" in out


def test_every_probed_host_is_recorded_for_the_receipt(table_repo: Path) -> None:
    """A refusal has to be auditable rather than believed, so every probed host
    lands in the trail that the receipt and the die() message both carry."""
    out = _pick(
        table_repo,
        load="h200=2.09,h201=3.10,h105=1.90",
        slot=_ALL_FREE,
        uv=_GOOD_UV,
    )
    for label in ("h200", "h201", "h101", "h105"):
        assert label in out


# =============================================================================
# The lock
# =============================================================================


def test_lock_is_exclusive(table_repo: Path, tmp_path: Path) -> None:
    wr = tmp_path / "wr"
    out = _driver(
        table_repo,
        f"prepush_lock_acquire {wr} && echo FIRST=ok\n"
        f'( PREPUSH_HELD_LOCK=""; prepush_lock_acquire {wr} && echo SECOND=ok || echo SECOND=blocked )\n',
    )
    assert "FIRST=ok" in out
    assert "SECOND=blocked" in out


def test_lock_is_reusable_after_release(table_repo: Path, tmp_path: Path) -> None:
    wr = tmp_path / "wr"
    out = _driver(
        table_repo,
        f"prepush_lock_acquire {wr} && echo FIRST=ok\n"
        "prepush_lock_release\n"
        f'( PREPUSH_HELD_LOCK=""; prepush_lock_acquire {wr} && echo SECOND=ok || echo SECOND=blocked )\n',
    )
    assert "FIRST=ok" in out
    assert "SECOND=ok" in out


def test_a_lock_whose_holder_is_dead_on_this_machine_is_reclaimed(
    table_repo: Path, tmp_path: Path
) -> None:
    """mkdir(2) is the lock primitive because flock(1) is absent on both Macs
    and its fd idiom needs `exec {fd}<>`, which bash 3.2 cannot parse. What
    mkdir lacks is auto-release on death, so a lock whose holder is provably
    gone is reclaimed -- without this one externally-SIGTERMed run (OMN-16713)
    wedges a host permanently."""
    wr = tmp_path / "wr"
    lockdir = wr / "LOCK"
    lockdir.mkdir(parents=True)
    host = subprocess.run(
        ["hostname", "-s"], capture_output=True, text=True, check=False
    ).stdout.strip()
    # pid 2^22 is above every default pid_max and is reliably absent.
    (lockdir / "holder").write_text(f"4194303 {host} 2026-01-01T00:00:00Z\n")
    out = _driver(
        table_repo,
        f"prepush_lock_acquire {wr} && echo RECLAIM=ok || echo RECLAIM=blocked",
    )
    assert "RECLAIM=ok" in out


def test_a_lock_held_by_a_live_process_is_not_reclaimed(
    table_repo: Path, tmp_path: Path
) -> None:
    wr = tmp_path / "wr"
    lockdir = wr / "LOCK"
    lockdir.mkdir(parents=True)
    host = subprocess.run(
        ["hostname", "-s"], capture_output=True, text=True, check=False
    ).stdout.strip()
    (lockdir / "holder").write_text(f"{os.getpid()} {host} 2026-01-01T00:00:00Z\n")
    out = _driver(
        table_repo,
        f"prepush_lock_acquire {wr} && echo RECLAIM=ok || echo RECLAIM=blocked",
    )
    assert "RECLAIM=blocked" in out


def test_a_lock_held_by_another_machine_is_never_reclaimed(
    table_repo: Path, tmp_path: Path
) -> None:
    """A pid from another host says nothing about whether a process here is
    alive, so a foreign holder is never reaped on a liveness check."""
    wr = tmp_path / "wr"
    lockdir = wr / "LOCK"
    lockdir.mkdir(parents=True)
    (lockdir / "holder").write_text("4194303 some-other-host 2026-01-01T00:00:00Z\n")
    out = _driver(
        table_repo,
        f"prepush_lock_acquire {wr} && echo RECLAIM=ok || echo RECLAIM=blocked",
    )
    assert "RECLAIM=blocked" in out


# =============================================================================
# Precedence and non-bypass invariants (static wiring)
# =============================================================================


def test_github_hosted_verification_is_tried_before_the_lab_leg() -> None:
    """OMN-16688's run is sha-pinned, green, full-suite shaped and re-derived
    live from the API with no file on disk to forge -- the hook's own comment
    calls it "strictly stronger evidence". The lab leg materializes the tree on
    another host and is admittedly weaker, so ordering it first would silently
    demote the strongest evidence the hook has."""
    text = HOOK.read_text(encoding="utf-8")
    start = text.index("guard_full_suite_host() {")
    guard = text[start:]
    for path_name, segment in (
        ("designated-host", guard[: guard.index("# Not a designated host")]),
        ("undesignated-host", guard[guard.index("# Not a designated host") :]),
    ):
        i_remote = segment.index("remote_full_suite_verified")
        i_lab = segment.index("dispatch_to_lab_host")
        i_grant = segment.index("consume_override_grant")
        assert i_remote < i_lab < i_grant, (
            f"{path_name} path: expected GitHub-hosted verify -> lab leg -> "
            "grant, got a different order"
        )


def test_a_remote_red_refuses_and_never_falls_through_to_a_grant() -> None:
    """A suite that genuinely failed on a designated host is a red gate, not a
    capacity problem. Letting it fall through to `consume_override_grant` would
    be a bypass wearing the word "fallback"."""
    text = HOOK.read_text(encoding="utf-8")
    start = text.index("dispatch_to_lab_host() {")
    body = text[start : text.index("guard_full_suite_host() {")]
    assert "3)" in body and "die " in body, (
        "expected the rc=3 (remote RED) branch of dispatch_to_lab_host to die"
    )
    red_branch = body[body.index("    3)") :]
    assert "die " in red_branch.split("esac")[0], (
        "the remote-RED branch must refuse, not return and fall through"
    )


def test_the_hook_introduces_no_new_bypass_env_knob() -> None:
    """Every knob added by OMN-16991 either routes work or makes the gate run
    MORE of it. None can make it accept less: the entry rejection of
    PREPUSH_ALLOW_* and the recursion sentinel are untouched."""
    text = HOOK.read_text(encoding="utf-8")
    assert "reject_inherited_env_overrides" in text
    assert 'if [ -n "${ONEX_PREPUSH_HOOK_ACTIVE:-}" ]; then' in text
    lib = LIB.read_text(encoding="utf-8")
    assert "PREPUSH_ALLOW" not in lib, (
        "the distribution library must not read any PREPUSH_ALLOW_* variable"
    )


def test_the_remote_command_rearms_both_guards() -> None:
    """ssh forwards neither the recursion sentinel nor the env scrub. Without
    re-arming, the remote repo's own suite -- which subprocesses this hook --
    takes FIRST-entry behavior there, resolves the selector, picks a host and
    ships another bundle: an unbounded DISTRIBUTED variant of the
    OMN-16425/OMN-16489 F-01 recursion (~9h03m, 44,064 tests)."""
    lib = LIB.read_text(encoding="utf-8")
    remote = lib[lib.index("cat > \"$runner\" <<'REMOTE'") : lib.index("\nREMOTE\n")]
    assert "export ONEX_PREPUSH_HOOK_ACTIVE=" in remote
    assert "PREPUSH_[A-Za-z0-9_]*" in remote, (
        "expected every PREPUSH_* name to be unset"
    )
    assert "unset ENABLE_SMART_TESTS" in remote


def test_the_verdict_is_read_from_a_marker_not_the_ssh_exit_code() -> None:
    """ssh returns 255 on transport failure (indistinguishable from a test
    failure) and any backgrounding wrapper returns 0 with nothing having run --
    a fail-OPEN shape. The marker binds the verdict to this tree and this argv;
    absence or mismatch is NO evidence."""
    lib = LIB.read_text(encoding="utf-8")
    assert 'marker="$(ssh' in lib
    assert '"$m_head" != "$head_sha"' in lib
    assert '"$m_argv" != "$argv_sha"' in lib
    assert "NO EVIDENCE" in lib


def test_a_shadow_host_verdict_never_authorizes() -> None:
    lib = LIB.read_text(encoding="utf-8")
    idx = lib.index('if [ "$PREPUSH_PICK_MODE" = "shadow" ]')
    branch = lib[idx : idx + 500]
    assert "return 1" in branch, (
        "a shadow host must fall through to the normal precedence, never authorize"
    )


def test_the_remote_wrapper_is_visible_to_the_201_queue_gate() -> None:
    """`.201`'s queue runner gates every lane on
    `ps ax | grep prepush_smart_tests.sh` ("covers foreign runs not launched
    through this queue"). Naming the remote wrapper to match makes a
    distributed run share that one mutex instead of becoming another foreign
    detached run -- the defect class OMN-16968 is open against."""
    lib = LIB.read_text(encoding="utf-8")
    assert 'runner="${localdir}/prepush_smart_tests.sh"' in lib
    assert "prepush_smart_tests.sh" in lib[lib.index("_PREPUSH_SLOT_PROBE_SH") :][:600]


def test_the_local_heavy_path_takes_the_host_lock() -> None:
    """OMN-16174: the local path took no lock of any kind, which is why five
    concurrent full suites once ran on one host with one taking 97+ minutes. It
    was the busiest path in the hook and the only unserialized one."""
    text = HOOK.read_text(encoding="utf-8")
    start = text.index("guard_full_suite_host() {")
    guard = text[start:]
    fit = guard[guard.index('if host_is_fit ""; then') :][:900]
    assert "prepush_lock_acquire" in fit
    assert "prepush_local_workroot" in fit


def test_the_escalation_argv_stays_a_superset_of_the_narrow_selection() -> None:
    """OMN-16825: the heavy call site runs $FULL_SUITE_TARGET **plus** the
    allowlisted service-free integration paths. Shipping only tests/unit/ to a
    remote host would silently drop tests/integration/chains/, a required Event
    Chain Gate surface, with no test firing."""
    lib = LIB.read_text(encoding="utf-8")
    argv = lib[lib.index("prepush_remote_argv() {") :]
    argv = argv[: argv.index("\n}\n")]
    assert "FULL_SUITE_TARGET" in argv
    assert "RUNNABLE_INTEGRATION_PATHS" in argv
    assert "PATHS" in argv


def test_the_dangling_runbook_pointer_is_gone() -> None:
    """The die() text cited docs/runbooks/200-build-lane-execution-pattern.md
    for months; that file has never existed in this repo (OMN-16446)."""
    for path in (
        HOOK,
        LIB,
        REPO_ROOT / "scripts" / "hooks" / "pytest_full_suite_host_guard.py",
    ):
        assert "200-build-lane-execution-pattern" not in path.read_text(
            encoding="utf-8"
        ), f"{path} still cites a runbook that does not exist"
    assert (REPO_ROOT / "docs" / "runbooks" / "lab-prepush-host-table.md").is_file()
