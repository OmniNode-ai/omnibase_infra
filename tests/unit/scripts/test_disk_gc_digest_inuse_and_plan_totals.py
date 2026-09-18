# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Digest-based in-use resolution and a citable dry-run total (OMN-16367).

WHY these exist
---------------
`disk-gc.sh` built its plan-time in-use set from::

    docker ps --all --format '{{.Image}}'

That field prints the image **NAME** for a name-referenced container and the id
only for an id-referenced one. Two lanes proved the failure on `.201` the same
day: a cross-reference built this way returned a **false zero on 29 of 30 rows**,
and the `201-data-breakdown-1045` report re-derived the whole inventory by
digest for exactly this reason.

Measured consequence, against a read-only capture of the live host (795 images,
61 distinct container image digests): the name-built plan selects **one image a
running container actually sits on** -- a dangling `postgres` layer, 611 MB --
and the digest-built plan selects zero.

The execution-time `docker ps -a --filter ancestor=...` re-check would very
likely have caught that one, but "a second check probably saves us" is a
works-by-convention surface, and the plan and its dry-run total are wrong in the
meantime. The dry-run total is what a host-rollout consent row cites, so a plan
that over-reports is a governance problem, not only a hygiene one.

The stability / judge / lakshman lanes make this sharper: their containers run on
digests **older than their own `:latest`**, so a name-based check protects the
tag while the running layer goes unprotected. Digest exclusion is the only sound
check for those lanes.

What this module pins
---------------------
1. An image referenced ONLY by digest is never planned for removal.
2. The plan carries a summed nominal byte total for what it would remove, so a
   dry run prints a number a consent row can quote.
3. The build-stamped tag shapes live in the committed keep-list, not in the
   script, so widening them is a config review rather than a code change.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"
_DISK_GC = _SCRIPTS / "disk-gc.sh"
_spec = importlib.util.spec_from_file_location(
    "disk_gc_plan", _SCRIPTS / "disk_gc_plan.py"
)
assert _spec and _spec.loader
disk_gc_plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(disk_gc_plan)

NOW = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)
AGENT = "onex-lab/omninode-runtime"


def _created(hours_ago: float) -> str:
    ts = NOW.timestamp() - hours_ago * 3600.0
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S +0000 UTC")


def _img(
    repo: str, tag: str, hours_ago: float, image_id: str, size: str = "3.66GB"
) -> dict[str, Any]:
    return {
        "ID": image_id,
        "Repository": repo,
        "Tag": tag,
        "CreatedAt": _created(hours_ago),
        "Size": size,
    }


KEEP_LIST: dict[str, Any] = {
    "keep_image_repos": ["omninode-runtime"],
    "keep_image_tags": ["latest", "stable", "rollback"],
    "protect_running": True,
    "superseded_image_keep_generations": 2,
    "min_age_days": 3,
    "generation_bounded_repos": ["onex-lab/"],
    "generation_keep": 2,
    "generation_min_age_hours": 6,
}


def _generations(count: int, start_hours: float = 8.0) -> list[dict[str, Any]]:
    return [
        _img(
            AGENT,
            f"20260918T{i:02d}0000Z-{i:08x}",
            start_hours + i,
            f"sha256:{i:064x}",
        )
        for i in range(count)
    ]


@pytest.mark.unit
class TestDigestOnlyReferenceIsHonoured:
    """A container referenced by NAME still pins a specific layer by digest."""

    def test_image_referenced_only_by_digest_is_kept(self) -> None:
        """RED before the fix when the in-use set is built from `{{.Image}}`.

        This is the stability-lane shape: the container runs an older digest while
        a `:latest` tag points somewhere else, so only the digest identifies the
        layer that must survive.
        """
        images = _generations(8)
        pinned = images[-1]
        plan = disk_gc_plan.build_plan(
            keep_list=KEEP_LIST,
            images=images,
            containers=[],
            inuse_refs={pinned["ID"]},  # digest only -- no repo:tag entry
            now=NOW,
        )
        assert pinned["ID"] not in set(plan["remove_image_ids"]), (
            "an image a container sits on by digest was planned for removal; "
            f"kept_reasons={plan['kept_reasons'].get(pinned['ID'])!r}"
        )

    def test_script_resolves_in_use_by_inspecting_container_image_digests(
        self, tmp_path: Path
    ) -> None:
        """`disk-gc.sh` must not build its in-use set from the name-printing field.

        AC falsifier: the script is RED if it ever goes back to deriving in-use
        refs from `docker ps --format '{{.Image}}'` alone, which is what produced
        a false zero on 29 of 30 rows on this host.
        """
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        argv_log = tmp_path / "argv.jsonl"
        argv_log.write_text("")
        fake = bin_dir / "docker"
        fake.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env python3
                import json, sys
                with open({str(argv_log)!r}, "a") as fh:
                    fh.write(json.dumps(sys.argv[1:]) + "\\n")
                args = sys.argv[1:]
                if args[:2] == ["ps", "-aq"] or args[:3] == ["ps", "-a", "-q"]:
                    print("cafe0001")
                    sys.exit(0)
                if args[:1] == ["inspect"]:
                    print("sha256:" + "9" * 64)
                    sys.exit(0)
                sys.exit(0)
                """
            )
        )
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        keep_list = tmp_path / "keep-list.yaml"
        keep_list.write_text(yaml.dump(KEEP_LIST))
        env = dict(os.environ)
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["HOME"] = str(tmp_path)

        subprocess.run(
            ["bash", str(_DISK_GC), "--keep-list", str(keep_list)],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            check=False,
        )
        recorded = [
            json.loads(line)
            for line in argv_log.read_text().splitlines()
            if line.strip()
        ]
        inspects = [a for a in recorded if a[:1] == ["inspect"]]
        assert inspects, (
            "disk-gc.sh never ran `docker inspect` to resolve a container's image "
            f"digest; recorded argv was {recorded!r}"
        )
        joined = " ".join(" ".join(a) for a in inspects)
        assert ".Image" in joined, (
            f"`docker inspect` was run but not for the .Image digest: {inspects!r}"
        )


@pytest.mark.unit
class TestPlanCarriesACitableTotal:
    """A dry run must print a number a consent row can quote."""

    def test_plan_reports_summed_nominal_bytes(self) -> None:
        images = _generations(6)  # 6 x 3.66GB, keep newest 2 -> 4 removed
        plan = disk_gc_plan.build_plan(
            keep_list=KEEP_LIST,
            images=images,
            containers=[],
            inuse_refs=set(),
            now=NOW,
        )
        assert "remove_nominal_bytes" in plan, (
            "the plan carries no size total; a dry run cannot cite a number"
        )
        assert len(plan["remove_image_ids"]) == 4
        expected = 4 * 3.66 * 1000**3
        assert abs(plan["remove_nominal_bytes"] - expected) < expected * 0.01

    def test_total_is_zero_when_nothing_is_removed(self) -> None:
        plan = disk_gc_plan.build_plan(
            keep_list=KEEP_LIST,
            images=_generations(2),
            containers=[],
            inuse_refs=set(),
            now=NOW,
        )
        assert plan["remove_image_ids"] == []
        assert plan["remove_nominal_bytes"] == 0

    def test_dry_run_prints_the_total(self, tmp_path: Path) -> None:
        """The number must reach the operator, not just the JSON."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        images = _generations(6)
        fake = bin_dir / "docker"
        fake.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env python3
                import json, sys
                args = sys.argv[1:]
                if args[:2] == ["image", "ls"]:
                    for i in {images!r}:
                        print(json.dumps(i))
                    sys.exit(0)
                if args[:1] == ["inspect"]:
                    sys.exit(0)
                sys.exit(0)
                """
            )
        )
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        keep_list = tmp_path / "keep-list.yaml"
        keep_list.write_text(yaml.dump(KEEP_LIST))
        env = dict(os.environ)
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["HOME"] = str(tmp_path)

        proc = subprocess.run(
            ["bash", str(_DISK_GC), "--keep-list", str(keep_list)],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            check=False,
        )
        log = tmp_path / ".local" / "log" / "onex" / "disk-gc.log"
        out = proc.stderr + (log.read_text() if log.exists() else "")
        assert "GB" in out and "would remove" in out.lower(), (
            f"the dry run printed no citable total. Output was:\n{out}"
        )


@pytest.mark.unit
class TestTagShapesAreConfigNotCode:
    """Widening what counts as a build stamp must be a config review."""

    def test_patterns_come_from_the_keep_list(self) -> None:
        """A caller-supplied pattern set must be honoured."""
        images = [
            _img(AGENT, "buildnum-4471", 40.0, "sha256:" + "a" * 64),
            *_generations(4),
        ]
        plan = disk_gc_plan.build_plan(
            keep_list={
                **KEEP_LIST,
                "generation_tag_patterns": [r"^buildnum-\d+$"],
            },
            images=images,
            containers=[],
            inuse_refs=set(),
            now=NOW,
        )
        # Only the custom shape is recognised now, so the four default-shaped
        # generations fall through and the single custom-shaped one is the whole
        # family -- inside the keep window, therefore kept.
        assert "sha256:" + "a" * 64 not in set(plan["remove_image_ids"])
        plan_many = disk_gc_plan.build_plan(
            keep_list={
                **KEEP_LIST,
                "generation_tag_patterns": [r"^buildnum-\d+$"],
            },
            images=[
                _img(AGENT, f"buildnum-{n}", 40.0 + n, f"sha256:{n:064x}")
                for n in range(5)
            ],
            containers=[],
            inuse_refs=set(),
            now=NOW,
        )
        assert len(plan_many["remove_image_ids"]) == 3, (
            "the configured pattern was not used to bound the family"
        )

    def test_shipped_keep_list_declares_the_patterns(self) -> None:
        keep_list = yaml.safe_load(
            (_REPO / "deploy" / "disk-gc" / "keep-list.yaml").read_text()
        )
        patterns = keep_list.get("generation_tag_patterns")
        assert patterns, (
            "generation_tag_patterns is absent from the committed keep-list; the "
            "tag shapes are still code-only"
        )
        # The shapes read off .201 must all still be covered by what ships.
        for sample in (
            "20260918T111918Z-e49dea8f",
            "ffd61901-20260918T111918Z",
            "sha-ffd6190",
            "dev-sha-ffd6190",
            "ffd6190171066f231bd6c6725fd53352d1222aa6",
        ):
            assert disk_gc_plan._is_build_stamped_tag(sample, patterns), (
                f"a tag shape measured on the host is no longer covered: {sample}"
            )

    @pytest.mark.parametrize("tag", ["dev", "main", "latest", "v1.2.3", "release-2026"])
    def test_meaningful_tags_match_no_shipped_pattern(self, tag: str) -> None:
        keep_list = yaml.safe_load(
            (_REPO / "deploy" / "disk-gc" / "keep-list.yaml").read_text()
        )
        patterns = keep_list["generation_tag_patterns"]
        assert not disk_gc_plan._is_build_stamped_tag(tag, patterns), (
            f"the shipped patterns treat the meaningful tag {tag!r} as disposable"
        )
