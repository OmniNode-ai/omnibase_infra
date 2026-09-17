# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev lane's ``onex-api`` pin is advanced by a script, never by hand [OMN-18113].

AC1 asks for a machine-checkable assertion that a sanctioned dev-lane refresh
path either advances ``ONEX_API_IMAGE`` to a build of the target ref, or declares
``onex-api`` out of its advancement scope. This file takes the first branch and
pins it from both ends:

* ``repoint_dev_lane_onex_api.py`` resolves, verifies and writes the pin, and
  refuses -- writing nothing -- on every ambiguous input (AC2, AC3).
* ``refresh_dev_lane.sh`` calls it and records the outcome in the receipt, so a
  refresh that left ``onex-api`` on a stale tag is a NAMED condition in the
  receipt rather than the silence measured on 2026-09-10 (AC4).
* ``refresh_stability_lane.sh`` does NOT call it, so this remains dev-lane-only
  (AC5).

Measured pre-fix behaviour, 2026-09-17 on the ``.201`` dev lane: ``ONEX_API_IMAGE``
was ``onex-lab/omnicloud-core:20260908T123804Z``, an unlabelled hand build from
2026-09-08, while ``onex-lab/omnicloud-core:f37261c2-20260917T050425Z`` -- built by
the lab-overlay applier that morning from ``omninode_infra`` ``origin/dev`` head --
sat resident on the same daemon, unreferenced. The image half was never missing;
the delivery half had no code at all.

The subprocess seams are faked rather than mocked at the library level: these
tests hand the script ``--docker`` and ``--git`` pointing at scripts this file
writes, so the argv the script actually builds is exercised. A mock of
``subprocess.run`` would pass even if the script assembled nonsense arguments.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_SCRIPT = _SCRIPT_DIR / "repoint_dev_lane_onex_api.py"
_REFRESH_DEV = _SCRIPT_DIR / "refresh_dev_lane.sh"
_REFRESH_STABILITY = _SCRIPT_DIR / "refresh_stability_lane.sh"
_LAB_OVERLAY = (
    _REPO_ROOT / "scripts" / "deploy-agent" / "deploy_agent" / "lab_overlay.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("repoint_dev_lane_onex_api", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    return _load()


# ---------------------------------------------------------------------------
# fakes for the two external commands
# ---------------------------------------------------------------------------


def _write_exec(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _fake_docker(
    tmp_path: Path,
    *,
    tags: list[str],
    resident: dict[str, str] | None = None,
    rc: int = 0,
) -> Path:
    """A docker stand-in that lists ``tags`` and inspects whatever is in ``resident``.

    ``resident`` maps a reference to a ``"<id>\\t<labels json>"`` line, so a tag
    can be listed and NOT resident -- which is the image-GC race the script has to
    refuse on rather than pin through.
    """
    payload = json.dumps({"tags": tags, "resident": resident or {}, "rc": rc})
    (tmp_path / "docker-fixture.json").write_text(payload, encoding="utf-8")
    return _write_exec(
        tmp_path / "fake-docker",
        f"""#!/usr/bin/env python3
import json, sys
fx = json.load(open({str(tmp_path / "docker-fixture.json")!r}))
argv = sys.argv[1:]
if argv[0] == "images":
    if fx["rc"]:
        sys.stderr.write("Cannot connect to the Docker daemon\\n")
        sys.exit(fx["rc"])
    print("\\n".join(fx["tags"]))
    sys.exit(0)
if argv[0] == "image" and argv[1] == "inspect":
    ref = argv[2]
    if ref not in fx["resident"]:
        sys.stderr.write("Error: No such image: %s\\n" % ref)
        sys.exit(1)
    print(fx["resident"][ref])
    sys.exit(0)
sys.exit(2)
""",
    )


def _fake_git(tmp_path: Path, *, known: dict[str, str], origin_dev: str | None) -> Path:
    payload = json.dumps({"known": known, "origin_dev": origin_dev})
    (tmp_path / "git-fixture.json").write_text(payload, encoding="utf-8")
    return _write_exec(
        tmp_path / "fake-git",
        f"""#!/usr/bin/env python3
import json, sys
fx = json.load(open({str(tmp_path / "git-fixture.json")!r}))
argv = sys.argv[1:]
rev = argv[-1]
if rev.startswith("origin/dev"):
    if fx["origin_dev"] is None:
        sys.stderr.write("unknown revision\\n")
        sys.exit(128)
    print(fx["origin_dev"])
    sys.exit(0)
short = rev.split("^")[0]
if short in fx["known"]:
    print(fx["known"][short])
    sys.exit(0)
sys.stderr.write("fatal: ambiguous argument '%s'\\n" % rev)
sys.exit(128)
""",
    )


def _clone(tmp_path: Path) -> Path:
    clone = tmp_path / "omninode_infra"
    (clone / ".git").mkdir(parents=True)
    return clone


def _env_file(tmp_path: Path, *, body: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(body, encoding="utf-8")
    return path


_FULL_SHA = "f37261c2ada3db7ca5eb194ee15507232f649029"
_NEW = "onex-lab/omnicloud-core:f37261c2-20260917T050425Z"
_OLDER = "onex-lab/omnicloud-core:5e4a8f84-20260916T173308Z"
_HAND = "onex-lab/omnicloud-core:20260908T123804Z"


def _run_script(
    tmp_path: Path,
    *,
    env_file: Path,
    clone: Path,
    docker: Path,
    git: Path,
    execute: bool = False,
    sha: str | None = None,
) -> tuple[int, dict[str, object]]:
    argv = [
        sys.executable,
        str(_SCRIPT),
        "--env-file",
        str(env_file),
        "--omninode-clone",
        str(clone),
        "--docker",
        str(docker),
        "--git",
        str(git),
    ]
    if execute:
        argv.append("--execute")
    if sha:
        argv += ["--sha", sha]
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    return completed.returncode, json.loads(completed.stdout)


# ---------------------------------------------------------------------------
# AC2 -- the build exists and this script names the same image the builder writes
# ---------------------------------------------------------------------------


def test_the_image_name_matches_the_builder_that_produces_it(mod: ModuleType) -> None:
    """The applier's ``API_IMAGE_NAME`` and this script's must be the same string.

    A rename on one side only would make the resolver find zero candidates
    forever, and a zero-candidate result reads exactly like "the applier has not
    run yet" -- a defect that never surfaces as an error.
    """
    source = _LAB_OVERLAY.read_text(encoding="utf-8")
    assert f'API_IMAGE_NAME = "{mod.API_IMAGE_NAME}"' in source


def test_the_builder_builds_the_api_image_from_the_ci_dockerfile(
    mod: ModuleType,
) -> None:
    """AC2's build half: the sanctioned builder already exists and is the CI one."""
    source = _LAB_OVERLAY.read_text(encoding="utf-8")
    assert 'API_DOCKERFILE = "docker/onex-api/Dockerfile"' in source
    assert 'API_CONTEXT = "docker/onex-api"' in source


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def test_newest_applier_built_tag_wins_over_an_older_one_and_over_a_hand_build(
    tmp_path: Path,
) -> None:
    docker = _fake_docker(
        tmp_path,
        tags=[_OLDER, _HAND, _NEW],
        resident={_NEW: f"sha256:abc\t{json.dumps(None)}"},
    )
    git = _fake_git(tmp_path, known={"f37261c2": _FULL_SHA}, origin_dev=_FULL_SHA)
    rc, out = _run_script(
        tmp_path,
        env_file=_env_file(tmp_path, body=f"A=1\nONEX_API_IMAGE={_HAND}\nB=2\n"),
        clone=_clone(tmp_path),
        docker=docker,
        git=git,
    )
    assert rc == 0
    assert out["result"] == "PLANNED"
    assert out["proposed"] == _NEW
    assert out["is_origin_dev_tip"] is True
    # The hand build is not a candidate: it does not carry a lineage.
    assert out["candidates_considered"] == 2
    assert out["tags_seen"] == 3


def test_sha_restriction_selects_that_lineage(tmp_path: Path) -> None:
    docker = _fake_docker(
        tmp_path,
        tags=[_OLDER, _NEW],
        resident={_OLDER: "sha256:old\tnull"},
    )
    git = _fake_git(
        tmp_path, known={"5e4a8f84": "5e4a8f84" + "0" * 32}, origin_dev=_FULL_SHA
    )
    rc, out = _run_script(
        tmp_path,
        env_file=_env_file(tmp_path, body=f"ONEX_API_IMAGE={_HAND}\n"),
        clone=_clone(tmp_path),
        docker=docker,
        git=git,
        sha="5e4a8f84",
    )
    assert rc == 0
    assert out["proposed"] == _OLDER
    assert out["is_origin_dev_tip"] is False


# ---------------------------------------------------------------------------
# refusals -- each one writes nothing
# ---------------------------------------------------------------------------


def test_zero_candidates_refuses_and_carries_a_positive_control(
    tmp_path: Path, mod: ModuleType
) -> None:
    """Rule 16: a zero that cannot be told apart from an unreadable daemon is not evidence."""
    env = _env_file(tmp_path, body=f"ONEX_API_IMAGE={_HAND}\n")
    before = env.read_text(encoding="utf-8")
    rc, out = _run_script(
        tmp_path,
        env_file=env,
        clone=_clone(tmp_path),
        docker=_fake_docker(tmp_path, tags=[_HAND], resident={}),
        git=_fake_git(tmp_path, known={}, origin_dev=_FULL_SHA),
        execute=True,
    )
    assert rc == mod.EXIT_REFUSED
    assert out["result"] == "REFUSED"
    assert "positive control" in str(out["reason"])
    assert "1 onex-lab/omnicloud-core tag(s)" in str(out["reason"])
    assert env.read_text(encoding="utf-8") == before


def test_an_unreadable_daemon_is_refused_as_such_and_not_as_an_absence(
    tmp_path: Path, mod: ModuleType
) -> None:
    rc, out = _run_script(
        tmp_path,
        env_file=_env_file(tmp_path, body=f"ONEX_API_IMAGE={_HAND}\n"),
        clone=_clone(tmp_path),
        docker=_fake_docker(tmp_path, tags=[], rc=1),
        git=_fake_git(tmp_path, known={}, origin_dev=None),
        execute=True,
    )
    assert rc == mod.EXIT_REFUSED
    assert "unreadable daemon" in str(out["reason"])


def test_a_listed_but_collected_image_is_refused_not_pinned(
    tmp_path: Path, mod: ModuleType
) -> None:
    """Image GC on this host can collect a tag between the listing and the write."""
    env = _env_file(tmp_path, body=f"ONEX_API_IMAGE={_HAND}\n")
    rc, out = _run_script(
        tmp_path,
        env_file=env,
        clone=_clone(tmp_path),
        docker=_fake_docker(tmp_path, tags=[_NEW], resident={}),
        git=_fake_git(tmp_path, known={"f37261c2": _FULL_SHA}, origin_dev=_FULL_SHA),
        execute=True,
    )
    assert rc == mod.EXIT_REFUSED
    assert "is not resident now" in str(out["reason"])
    assert env.read_text(encoding="utf-8") == f"ONEX_API_IMAGE={_HAND}\n"


def test_a_tag_whose_sha_is_not_a_commit_is_refused(
    tmp_path: Path, mod: ModuleType
) -> None:
    rc, out = _run_script(
        tmp_path,
        env_file=_env_file(tmp_path, body=f"ONEX_API_IMAGE={_HAND}\n"),
        clone=_clone(tmp_path),
        docker=_fake_docker(tmp_path, tags=[_NEW], resident={_NEW: "sha256:abc\tnull"}),
        git=_fake_git(tmp_path, known={}, origin_dev=_FULL_SHA),
        execute=True,
    )
    assert rc == mod.EXIT_REFUSED
    assert "does not resolve" in str(out["reason"])


def test_a_label_disagreeing_with_the_tag_is_refused(
    tmp_path: Path, mod: ModuleType
) -> None:
    labels = json.dumps({mod.REVISION_LABEL: "deadbeef" + "0" * 32})
    rc, out = _run_script(
        tmp_path,
        env_file=_env_file(tmp_path, body=f"ONEX_API_IMAGE={_HAND}\n"),
        clone=_clone(tmp_path),
        docker=_fake_docker(
            tmp_path, tags=[_NEW], resident={_NEW: f"sha256:abc\t{labels}"}
        ),
        git=_fake_git(tmp_path, known={"f37261c2": _FULL_SHA}, origin_dev=_FULL_SHA),
        execute=True,
    )
    assert rc == mod.EXIT_REFUSED
    assert "disagreeing provenance" in str(out["reason"])


def test_an_agreeing_label_is_accepted(tmp_path: Path) -> None:
    """The label is forward-compatible provenance, not a second source of truth."""
    labels = json.dumps({"org.opencontainers.image.revision": _FULL_SHA})
    rc, out = _run_script(
        tmp_path,
        env_file=_env_file(tmp_path, body=f"ONEX_API_IMAGE={_HAND}\n"),
        clone=_clone(tmp_path),
        docker=_fake_docker(
            tmp_path, tags=[_NEW], resident={_NEW: f"sha256:abc\t{labels}"}
        ),
        git=_fake_git(tmp_path, known={"f37261c2": _FULL_SHA}, origin_dev=_FULL_SHA),
    )
    assert rc == 0
    assert out["revision_label"] == _FULL_SHA


def test_a_missing_pin_key_is_refused_rather_than_appended(
    tmp_path: Path, mod: ModuleType
) -> None:
    """Rule 8. Deciding this lane should have a pin is not this script's call."""
    env = _env_file(tmp_path, body="A=1\nB=2\n")
    rc, out = _run_script(
        tmp_path,
        env_file=env,
        clone=_clone(tmp_path),
        docker=_fake_docker(tmp_path, tags=[_NEW], resident={_NEW: "sha256:abc\tnull"}),
        git=_fake_git(tmp_path, known={"f37261c2": _FULL_SHA}, origin_dev=_FULL_SHA),
        execute=True,
    )
    assert rc == mod.EXIT_REFUSED
    assert "carries no ONEX_API_IMAGE= line" in str(out["reason"])
    assert env.read_text(encoding="utf-8") == "A=1\nB=2\n"


def test_a_duplicated_pin_key_is_refused(tmp_path: Path, mod: ModuleType) -> None:
    """A sourcing shell takes the LAST assignment, so a single-line rewrite lies."""
    body = f"ONEX_API_IMAGE={_HAND}\nX=1\nONEX_API_IMAGE={_OLDER}\n"
    env = _env_file(tmp_path, body=body)
    rc, out = _run_script(
        tmp_path,
        env_file=env,
        clone=_clone(tmp_path),
        docker=_fake_docker(tmp_path, tags=[_NEW], resident={_NEW: "sha256:abc\tnull"}),
        git=_fake_git(tmp_path, known={"f37261c2": _FULL_SHA}, origin_dev=_FULL_SHA),
        execute=True,
    )
    assert rc == mod.EXIT_REFUSED
    assert "2 ONEX_API_IMAGE= lines" in str(out["reason"])
    assert env.read_text(encoding="utf-8") == body


def test_a_missing_env_file_is_refused(tmp_path: Path, mod: ModuleType) -> None:
    rc, out = _run_script(
        tmp_path,
        env_file=tmp_path / "absent.env",
        clone=_clone(tmp_path),
        docker=_fake_docker(tmp_path, tags=[_NEW], resident={_NEW: "sha256:abc\tnull"}),
        git=_fake_git(tmp_path, known={"f37261c2": _FULL_SHA}, origin_dev=_FULL_SHA),
        execute=True,
    )
    assert rc == mod.EXIT_REFUSED
    assert "does not exist" in str(out["reason"])


def test_a_clone_that_is_not_a_clone_is_refused(
    tmp_path: Path, mod: ModuleType
) -> None:
    rc, out = _run_script(
        tmp_path,
        env_file=_env_file(tmp_path, body=f"ONEX_API_IMAGE={_HAND}\n"),
        clone=tmp_path / "not-a-clone",
        docker=_fake_docker(tmp_path, tags=[_NEW], resident={_NEW: "sha256:abc\tnull"}),
        git=_fake_git(tmp_path, known={"f37261c2": _FULL_SHA}, origin_dev=_FULL_SHA),
        execute=True,
    )
    assert rc == mod.EXIT_REFUSED
    assert "is not a git clone" in str(out["reason"])


def test_both_paths_are_required_arguments(tmp_path: Path, mod: ModuleType) -> None:
    """Neither has a default, so neither can be silently wrong (rule 8)."""
    for missing in ("--env-file", "--omninode-clone"):
        argv = [
            sys.executable,
            str(_SCRIPT),
            "--env-file",
            str(tmp_path / ".env"),
            "--omninode-clone",
            str(tmp_path),
        ]
        index = argv.index(missing)
        del argv[index : index + 2]
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
        assert completed.returncode == mod.EXIT_USAGE
        assert missing in completed.stderr


# ---------------------------------------------------------------------------
# the write
# ---------------------------------------------------------------------------


def test_plan_mode_writes_nothing(tmp_path: Path) -> None:
    body = f"A=1\nONEX_API_IMAGE={_HAND}\nB=2\n"
    env = _env_file(tmp_path, body=body)
    rc, out = _run_script(
        tmp_path,
        env_file=env,
        clone=_clone(tmp_path),
        docker=_fake_docker(tmp_path, tags=[_NEW], resident={_NEW: "sha256:abc\tnull"}),
        git=_fake_git(tmp_path, known={"f37261c2": _FULL_SHA}, origin_dev=_FULL_SHA),
    )
    assert rc == 0
    assert out["result"] == "PLANNED"
    assert out["tag_advanced"] is False
    assert env.read_text(encoding="utf-8") == body


def test_execute_changes_exactly_one_line_and_keeps_a_backup(tmp_path: Path) -> None:
    body = f"A=1\nONEX_API_IMAGE={_HAND}\nB=2\nC=3\n"
    env = _env_file(tmp_path, body=body)
    rc, out = _run_script(
        tmp_path,
        env_file=env,
        clone=_clone(tmp_path),
        docker=_fake_docker(tmp_path, tags=[_NEW], resident={_NEW: "sha256:abc\tnull"}),
        git=_fake_git(tmp_path, known={"f37261c2": _FULL_SHA}, origin_dev=_FULL_SHA),
        execute=True,
    )
    assert rc == 0
    assert out["result"] == "WRITTEN"
    assert out["tag_advanced"] is True
    assert out["pin_before"] == _HAND
    assert out["pin_after"] == _NEW
    after = env.read_text(encoding="utf-8")
    assert after == f"A=1\nONEX_API_IMAGE={_NEW}\nB=2\nC=3\n"
    backup = Path(str(out["backup"]))
    assert backup.is_file()
    assert backup.read_text(encoding="utf-8") == body


def test_a_second_run_is_a_no_op_and_takes_no_backup(tmp_path: Path) -> None:
    """Idempotence matters: this runs on every governed refresh."""
    env = _env_file(tmp_path, body=f"ONEX_API_IMAGE={_NEW}\n")
    docker = _fake_docker(tmp_path, tags=[_NEW], resident={_NEW: "sha256:abc\tnull"})
    git = _fake_git(tmp_path, known={"f37261c2": _FULL_SHA}, origin_dev=_FULL_SHA)
    rc, out = _run_script(
        tmp_path,
        env_file=env,
        clone=_clone(tmp_path),
        docker=docker,
        git=git,
        execute=True,
    )
    assert rc == 0
    assert out["result"] == "UNCHANGED"
    assert out["tag_advanced"] is False
    assert out["backup"] is None
    assert not list(env.parent.glob(".env.bak.*"))


def test_a_readback_that_fails_restores_the_file(
    tmp_path: Path, mod: ModuleType
) -> None:
    """The operator env file carries every other lane's configuration.

    A write that touched a second line is not recoverable from a receipt, so the
    script restores from its own backup and refuses rather than reporting success.
    """
    body = f"A=1\nONEX_API_IMAGE={_HAND}\n"
    env = _env_file(tmp_path, body=body)
    lines = env.read_text(encoding="utf-8").splitlines(keepends=True)

    original_write = Path.write_text
    calls = {"n": 0}

    def sabotaged(self: Path, data: str, *args: object, **kwargs: object) -> int:
        calls["n"] += 1
        if self == env and calls["n"] == 1:
            data = data + "SNEAKY=1\n"
        return original_write(self, data, *args, **kwargs)  # type: ignore[arg-type]

    monkey = pytest.MonkeyPatch()
    monkey.setattr(Path, "write_text", sabotaged)
    try:
        with pytest.raises(mod.RepointRefusalError) as excinfo:
            mod.write_pin(
                env, lines=lines, index=1, reference=_NEW, stamp="20260917T000000Z"
            )
    finally:
        monkey.undo()

    assert "readback refused the write" in str(excinfo.value)
    assert env.read_text(encoding="utf-8") == body


# ---------------------------------------------------------------------------
# AC4 / AC5 -- wiring, and the lane boundary
# ---------------------------------------------------------------------------


def test_the_dev_refresh_calls_the_repoint_script(mod: ModuleType) -> None:
    """AC1/AC3: the sanctioned dev-lane refresh path advances the pin itself."""
    source = _REFRESH_DEV.read_text(encoding="utf-8")
    assert "repoint_dev_lane_onex_api.py" in source


def test_the_dev_refresh_receipt_names_onex_api_distinctly(mod: ModuleType) -> None:
    """AC4: recreated and tag_advanced are recorded separately from the other services.

    The 2026-09-10 measurement that opened this ticket was that the string
    ``onex-api`` appeared NOWHERE in a full refresh log. These four keys are what
    makes the same run's outcome readable without reading the log at all.
    """
    source = _REFRESH_DEV.read_text(encoding="utf-8")
    for key in (
        "onex_api:",
        "pin_before:",
        "pin_after:",
        "tag_advanced:",
        "recreated:",
    ):
        assert key in source, f"receipt is missing {key}"


def test_the_stability_refresh_does_not_repoint(mod: ModuleType) -> None:
    """AC5: prod and stability-test ``onex-api`` handling is unchanged."""
    assert "repoint_dev_lane_onex_api.py" not in _REFRESH_STABILITY.read_text(
        encoding="utf-8"
    )


def test_the_repoint_names_the_dev_lane_env_key_and_nothing_broader(
    mod: ModuleType,
) -> None:
    """The script rewrites one key. A second key here would widen the blast radius."""
    source = _SCRIPT.read_text(encoding="utf-8")
    assert mod.PIN_KEY == "ONEX_API_IMAGE"
    assert "ONEX_CLOUD_MIGRATE_IMAGE" not in source


def test_the_negative_assertions_above_are_not_vacuous() -> None:
    """Positive control for the two absence assertions in this file.

    ``test_the_stability_refresh_does_not_repoint`` and the
    ``ONEX_CLOUD_MIGRATE_IMAGE`` assertion would both pass against an empty file.
    This proves the files are readable and non-trivial first.
    """
    assert len(_REFRESH_STABILITY.read_text(encoding="utf-8")) > 1000
    assert "refresh_stability_lane" in _REFRESH_STABILITY.read_text(encoding="utf-8")
    assert len(_SCRIPT.read_text(encoding="utf-8")) > 1000


def test_the_builder_stamps_the_omninode_sha_as_an_oci_label(mod: ModuleType) -> None:
    """Provenance that survives a retag, cross-checked by the script above.

    The image the lane ran on 2026-09-17 carried NO labels at all, so the only
    provenance was its tag -- and its tag was a bare timestamp naming no commit.
    """
    source = _LAB_OVERLAY.read_text(encoding="utf-8")
    assert f'API_REVISION_LABEL = "{mod.REVISION_LABEL}"' in source
    assert "labels={" in source
    assert "API_REVISION_LABEL: manifest_sha" in source


def test_build_and_import_puts_labels_on_the_argv_it_builds() -> None:
    """The label has to reach ``docker build``, not just a keyword argument.

    Read from the builder's own source rather than by invoking it: this function
    shells out to docker and containerd, and a test that stubbed both would be
    asserting the stub. What is checked here is the one line that can be wrong.
    """
    source = _LAB_OVERLAY.read_text(encoding="utf-8")
    assert 'argv += ["--label", f"{name}={value}"]' in source


def test_the_script_is_executable_and_self_documenting() -> None:
    assert os.access(_SCRIPT, os.X_OK), "the refresh calls this script by path"
    assert _SCRIPT.read_text(encoding="utf-8").startswith("#!/usr/bin/env python3")
