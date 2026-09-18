# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Builder-cache size-cap tests for `disk-gc.sh` (OMN-16367).

WHY these exist
---------------
On 2026-09-18 `docker buildx du` on .201 reported **749.7 GB of build cache,
100% reclaimable, across 9,183 records**, while the hourly `onex-disk-gc.timer`
reclaimed *kilobytes* per pass::

    [2026-09-18T08:06:55Z] [disk-gc] Pruning builder cache older than 3d
    Total:	0B
    Total:	24.73kB
    Total:	20.48kB

The prune was::

    docker builder prune --force --filter "until=${MIN_AGE_DAYS}h0m0s"

Two defects, one decisive:

1. **Units bug** — ``MIN_AGE_DAYS=3`` was interpolated into an *hours* filter
   (``until=3h0m0s``) while the log line said "3d". Errs aggressive, so it is
   not the cause, but a variable whose name ends ``_DAYS`` feeding a string
   ending ``h0m0s`` is a latent trap.
2. **The cause** — ``until=`` filters on **last accessed**, not on creation.
   Every large record (the 1.47 GB ``/app/.venv`` layer, the torch/uv base) is
   served as a cache hit by the next proof build minutes later, which refreshes
   its timestamp. The big records are therefore *never* eligible and an
   age-filtered prune is **structurally incapable of bounding this cache**.
   It needs a size ceiling, not an age filter.

These tests pin the size cap so a revert to an age-only bound is a RED test
rather than a silently-kilobyte-reclaiming timer nobody reads.

Flag choice, verified live on .201 (2026-09-18, Docker 29.2.1 / buildx v0.31.1)
-------------------------------------------------------------------------------
``docker builder prune --help`` offers ``--max-used-space``, ``--min-free-space``
and ``--reserved-space``. ``--keep-storage`` still parses but emits
``Flag --keep-storage has been deprecated, keep-storage flag has been changed to
reserved-space`` — i.e. it maps to the **floor** (space always *kept*), which is
the opposite of the ceiling we need. ``--max-used-space`` is the ceiling and is
what the script must emit.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"
_DISK_GC = _SCRIPTS / "disk-gc.sh"

# docker CreatedAt format the planner parses; old enough to never be a candidate.
_OLD_CREATED = "2026-01-01 00:00:00 +0000 UTC"


def _write_fake_docker(bin_dir: Path, argv_log: Path, reclaimed: str) -> None:
    """A fake `docker` that records every argv and fakes a prune reclaim total.

    Every invocation appends its full argv as one JSON line to ``argv_log`` so a
    test can assert on exactly what the script asked docker to do. The `builder
    prune` branch prints a realistic ``Total:`` line so the reclaimed-bytes log
    assertion has something to find.
    """
    fake = bin_dir / "docker"
    fake.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import json, sys

            ARGV_LOG = {str(argv_log)!r}
            RECLAIMED = {reclaimed!r}

            args = sys.argv[1:]
            with open(ARGV_LOG, "a") as fh:
                fh.write(json.dumps(args) + "\\n")

            # Inventory probes: empty host, so the plan removes no image and no
            # container and the run exercises the builder-cache path only.
            if args[:2] == ["image", "ls"]:
                sys.exit(0)
            if args[:2] == ["ps", "--all"]:
                sys.exit(0)
            if args[:2] == ["ps", "-a"]:
                sys.exit(0)

            if args[:2] == ["builder", "prune"]:
                # Reject the deprecated floor flag the way real buildx >= 0.31 does
                # for an unknown flag, so a test can prove the fallback chain.
                if "--keep-storage" in args:
                    sys.stderr.write("unknown flag: --keep-storage\\n")
                    sys.exit(125)
                print("Total:\\t" + RECLAIMED)
                sys.exit(0)

            sys.exit(0)
            """
        )
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run(
    tmp_path: Path,
    *,
    keep_list_extra: dict[str, object] | None = None,
    reclaimed: str = "512.3MB",
) -> tuple[subprocess.CompletedProcess[str], str, list[list[str]]]:
    """Run `disk-gc.sh --execute` against the fake docker.

    Returns (process, combined stderr+log text, recorded docker argv list).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    argv_log = tmp_path / "docker_argv.jsonl"
    argv_log.write_text("")
    _write_fake_docker(bin_dir, argv_log, reclaimed)

    keep_list_data: dict[str, object] = {
        "keep_image_repos": ["myrepo"],
        "keep_image_tags": [],
        "protect_running": True,
        "superseded_image_keep_generations": 0,
        "min_age_days": 3,
    }
    if keep_list_extra:
        keep_list_data.update(keep_list_extra)

    keep_list = tmp_path / "keep-list.yaml"
    keep_list.write_text(yaml.dump(keep_list_data))

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)

    proc = subprocess.run(
        ["bash", str(_DISK_GC), "--execute", "--keep-list", str(keep_list)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )

    log_file = tmp_path / ".local" / "log" / "onex" / "disk-gc.log"
    log_text = log_file.read_text() if log_file.exists() else ""

    recorded = [
        json.loads(line) for line in argv_log.read_text().splitlines() if line.strip()
    ]
    return proc, proc.stderr + log_text, recorded


def _builder_prunes(recorded: list[list[str]]) -> list[list[str]]:
    return [a for a in recorded if a[:2] == ["builder", "prune"]]


@pytest.mark.unit
class TestBuilderCacheSizeCap:
    """AC2 — the prune is size-bounded, not age-bounded."""

    def test_builder_prune_emits_a_size_cap_flag(self, tmp_path: Path) -> None:
        """The primary prune must carry `--max-used-space <cap>`.

        RED against the pre-OMN-16367 script, whose only bound was
        `--filter until=...`. This is AC2's falsifier: the test fails if the
        command ever reverts to an age-only bound.
        """
        proc, _, recorded = _run(tmp_path)
        assert proc.returncode == 0, proc.stderr

        prunes = _builder_prunes(recorded)
        assert prunes, "disk-gc.sh issued no `docker builder prune` at all"

        primary = prunes[0]
        assert "--max-used-space" in primary, (
            "the primary builder prune carries no size cap; emitted argv was "
            f"{primary!r}. An age-only bound cannot shrink a cache whose large "
            "records are refreshed by every cache hit."
        )
        cap = primary[primary.index("--max-used-space") + 1]
        assert re.fullmatch(r"\d+(\.\d+)?(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)", cap), (
            f"size cap {cap!r} is not a docker-parseable size string"
        )

    def test_primary_prune_passes_all(self, tmp_path: Path) -> None:
        """`--all` is load-bearing on a containerd-snapshotter host, not aggression.

        .201 runs `io.containerd.snapshotter.v1`, so build-cache records share the
        content store with image layers and a DEFAULT builder prune excludes every
        record an existing image references. Measured live 2026-09-18:
        `docker builder prune --force --max-used-space 200GB` WITHOUT `--all`
        reclaimed **0B at exit 0**, twice, while `buildx du` reported 655-678 GB
        reclaimable (ROLLING_WORK_LEDGER.md:3932, :3933).

        A correct cap flag without `--all` is a silent no-op on this host -- the
        worst shape a disk-pressure remedy can take, because a caller reading only
        the exit status records a successful prune that freed nothing. This test
        exists so a future "that looks too aggressive" edit is RED.
        """
        _, _, recorded = _run(tmp_path)
        primary = _builder_prunes(recorded)[0]
        assert "--all" in primary, (
            "the capped prune omits --all and is therefore a no-op on a "
            f"containerd-snapshotter host; emitted argv was {primary!r}"
        )

    def test_primary_prune_is_not_also_age_filtered(self, tmp_path: Path) -> None:
        """An `until=` filter on the SAME call would re-break the cap.

        `until=` restricts which records are *eligible*; the cap sets the target.
        Passing both means the cap can never be reached whenever the oversized
        records are the recently-accessed ones -- which is exactly the observed
        failure. The age filter is retained only as a FALLBACK for a docker too
        old to know the cap flag, never as an AND-constraint on the capped call.
        """
        _, _, recorded = _run(tmp_path)
        primary = _builder_prunes(recorded)[0]
        joined = " ".join(primary)
        assert "until=" not in joined, (
            "the size-capped prune is also age-filtered, which restores the bug: "
            f"{primary!r}"
        )

    def test_cap_is_read_from_config_not_hardcoded(self, tmp_path: Path) -> None:
        """The cap comes from one keep-list value, per the script's own doctrine.

        disk-gc.sh's header states the keep-list is the single source of truth and
        that "nothing is hardcoded in the script".
        """
        _, _, recorded = _run(
            tmp_path, keep_list_extra={"builder_cache_max_size": "37GB"}
        )
        primary = _builder_prunes(recorded)[0]
        assert primary[primary.index("--max-used-space") + 1] == "37GB", (
            f"configured cap was not honoured; emitted argv was {primary!r}"
        )

    def test_default_cap_applies_when_config_omits_it(self, tmp_path: Path) -> None:
        """A keep-list with no cap key still gets a bounded prune (default 100GB)."""
        _, _, recorded = _run(tmp_path)
        primary = _builder_prunes(recorded)[0]
        assert primary[primary.index("--max-used-space") + 1] == "100GB"

    def test_reclaimed_bytes_are_logged(self, tmp_path: Path) -> None:
        """The reclaim total must reach the log, so the live falsifier is readable.

        AC2's live falsifier is "three consecutive disk-gc.log entries reporting
        Total: <1MB". That is only checkable if the script records the figure on
        its own tagged line rather than letting docker's bare `Total:` blend into
        the log.
        """
        _, output, _ = _run(tmp_path, reclaimed="512.3MB")
        assert "512.3MB" in output, "prune reclaim total never reached the log"
        assert re.search(r"\[disk-gc\].*reclaimed.*512\.3MB", output, re.IGNORECASE), (
            "reclaimed bytes are not on a disk-gc-tagged log line; the hourly "
            f"reclaim cannot be audited from the log alone. Log was:\n{output}"
        )


@pytest.mark.unit
class TestAgeUnitsBugCannotReturn:
    """AC3 — the units bug is fixed and pinned.

    AC3's falsifier, implemented literally: a test that reads the script and
    fails when a variable whose name ends `_DAYS` is interpolated into a string
    ending `h0m0s`.
    """

    # A *_DAYS variable interpolated into an hours-unit duration.
    _DAYS_INTO_HOURS = re.compile(r"\$\{?[A-Za-z_]*_DAYS\}?[^\"']*h0m0s")
    _HOURS_INTO_HOURS = re.compile(r"\$\{?[A-Za-z_]*_HOURS\}?[^\"']*h0m0s")

    @staticmethod
    def _code_lines() -> list[str]:
        """Executable lines of disk-gc.sh, comments stripped.

        Comment prose is excluded deliberately. These guards read source text, so
        a comment *explaining* the historical bug would otherwise trip the guard
        that exists to catch the bug -- the same failure mode as a gate firing on
        documentation about the gate. `_positive_control` below is what proves
        the detector still bites.
        """
        return [
            line.strip()
            for line in _DISK_GC.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def test_detector_matches_the_historical_buggy_line(self) -> None:
        """Positive control: an empty offender list must mean absence, not a dud regex.

        This is the exact line that shipped before OMN-16367.
        """
        historical = (
            'docker builder prune --force --filter "until=${MIN_AGE_DAYS}h0m0s"'
        )
        assert self._DAYS_INTO_HOURS.search(historical), (
            "the offender detector no longer matches the very line it was written "
            "for; a zero-offender result from it would be meaningless"
        )

    def test_no_days_variable_feeds_an_hours_duration(self) -> None:
        offenders = [
            line for line in self._code_lines() if self._DAYS_INTO_HOURS.search(line)
        ]
        assert not offenders, (
            "a *_DAYS variable is interpolated into an hours-unit duration: a "
            "min-age expressed in DAYS silently means that many HOURS while the "
            f"log reports days. Offending line(s): {offenders!r}"
        )

    def test_age_fallback_filter_uses_an_hours_named_variable(self) -> None:
        """Whatever feeds `h0m0s` must be named in hours, so name and unit agree."""
        hours_filters = [line for line in self._code_lines() if "h0m0s" in line]
        assert hours_filters, "the age fallback filter disappeared entirely"
        for line in hours_filters:
            assert self._HOURS_INTO_HOURS.search(line), (
                f"an hours-unit duration is fed by a non-_HOURS variable: {line!r}"
            )

    def test_log_line_and_command_agree_on_units(self, tmp_path: Path) -> None:
        """The log must not claim days while the command means hours."""
        _, output, _ = _run(tmp_path)
        # min_age_days: 3 -> the fallback bound is 72 hours. If the log mentions a
        # bare "3d" alongside a 3-hour command, the 2026-09-18 log line is back.
        assert "older than 3d" not in output, (
            "the log still reports '3d' for what the fallback expresses in hours"
        )


@pytest.mark.unit
class TestDaemonBuilderGcFragment:
    """AC1 — the daemon-level ceiling exists as a reviewable, non-drifting artifact.

    The fragment is deliberately NOT applied by anything in this repo: installing
    it edits `/etc/docker/daemon.json` and needs a dockerd reload, which bounces
    every lane on the host. These tests cover what CI *can* prove -- that the
    committed policy is well-formed, uses the ceiling vocabulary rather than the
    floor vocabulary, and does not drift from the hourly GC's own cap.
    """

    _FRAGMENT = _REPO / "deploy" / "disk-gc" / "daemon-builder-gc.json"
    _KEEP_LIST = _REPO / "deploy" / "disk-gc" / "keep-list.yaml"

    def _policy(self) -> dict[str, object]:
        return json.loads(self._FRAGMENT.read_text())

    def test_fragment_is_valid_applyable_daemon_json(self) -> None:
        """dockerd rejects unknown top-level keys, so the fragment carries none.

        A fragment that cannot be merged into daemon.json as-is is documentation
        pretending to be configuration.
        """
        doc = self._policy()
        assert set(doc) == {"builder"}, (
            "the fragment carries top-level keys dockerd would reject "
            f"(e.g. a comment key): {sorted(doc)}"
        )
        rules = doc["builder"]["gc"]["policy"]  # type: ignore[index,call-overload]
        assert isinstance(rules, list) and rules, "no GC policy rules declared"

    def test_catch_all_rule_sets_a_ceiling_not_only_a_floor(self) -> None:
        """`reservedSpace` alone prunes toward nothing -- measured 0B on .201.

        `reservedSpace` (and its `keepStorage` alias) is a FLOOR. Without
        `maxUsedSpace` or `minFreeSpace`, BuildKit has nothing to prune toward and
        reclaims nothing while exiting 0. See ROLLING_WORK_LEDGER.md:3921/:3932.
        """
        rules = self._policy()["builder"]["gc"]["policy"]  # type: ignore[index,call-overload]
        catch_all = [r for r in rules if r.get("all") is True]
        assert catch_all, (
            "no catch-all (`all: true`) rule; unmatched cache is unbounded"
        )
        for rule in catch_all:
            assert "maxUsedSpace" in rule, (
                "the catch-all rule declares no ceiling, only a floor; this is the "
                f"configuration equivalent of the 0B no-op prune: {rule!r}"
            )

    def test_fragment_never_uses_the_deprecated_floor_vocabulary(self) -> None:
        """`keepStorage` is an alias for the FLOOR and must not appear as the cap."""
        raw = self._FRAGMENT.read_text()
        assert "keepStorage" not in raw and "keepBytes" not in raw, (
            "the fragment is written in the deprecated keep-storage vocabulary, "
            "which aliases to reservedSpace (a floor) and cannot bound the cache"
        )

    def test_daemon_ceiling_matches_the_hourly_gc_cap(self) -> None:
        """The standing ceiling and the scheduled enforcement are one number.

        The README promises these stay in step; this is what makes that
        mechanical rather than aspirational.
        """
        rules = self._policy()["builder"]["gc"]["policy"]  # type: ignore[index,call-overload]
        catch_all = next(r for r in rules if r.get("all") is True)
        keep_list = yaml.safe_load(self._KEEP_LIST.read_text())
        assert catch_all["maxUsedSpace"] == keep_list["builder_cache_max_size"], (
            "daemon-builder-gc.json maxUsedSpace "
            f"({catch_all['maxUsedSpace']}) has drifted from keep-list.yaml "
            f"builder_cache_max_size ({keep_list['builder_cache_max_size']}); "
            "change both in the same PR"
        )
