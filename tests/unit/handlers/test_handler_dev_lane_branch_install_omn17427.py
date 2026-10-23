# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``handler_dev_lane_branch_install``: a declared record the dev deploy agent re-applies (OMN-17427).

On 2026-10-22 a fix installed into a running dev-lane container from its branch was
dropped twice by the next dev refresh: the hook-event feed fix (omnimarket#4304,
omnimarket#4308, omnibase_infra#5269) around 03:08Z and the consumer-flow writer
(omnimarket#4312, sha256 17d6b307) at the 11:18Z refresh. A fake Docker holds one
dev-lane runtime container whose files come from its image; ``recreate`` replays
the refresh. These tests pin that a declared install is detected as missing after
the recreate, put back, restarted once and read back at its recorded sha256, that
it is dropped on merge or expiry, and that nothing is ever applied off the dev lane.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from omnibase_infra.enums.enum_dev_lane_branch_install_file_state import (
    EnumDevLaneBranchInstallFileState,
)
from omnibase_infra.enums.enum_dev_lane_branch_install_release_reason import (
    EnumDevLaneBranchInstallReleaseReason,
)
from omnibase_infra.enums.enum_dev_lane_branch_install_state import (
    EnumDevLaneBranchInstallState,
)
from omnibase_infra.handlers import handler_dev_lane_branch_install as bi
from omnibase_infra.handlers.handler_dev_lane_branch_install import (
    DevLaneBranchInstallRefusedError,
    DevLaneBranchInstallStore,
    HandlerDevLaneBranchInstall,
    Runner,
    git_merged_probe,
    register,
    release,
    status_payload,
)
from omnibase_infra.models.model_dev_lane_branch_install import (
    ModelDevLaneBranchInstall,
)

pytestmark = pytest.mark.unit

PROJECT = "omnibase-infra"
SERVICE = "omninode-runtime"
WRITER = (
    "/app/.venv/lib/python3.12/site-packages/omnimarket/nodes/"
    "node_projection_consumer_flow/handlers/handler_consumer_flow_runner.py"
)
FEED = (
    "/app/.venv/lib/python3.12/site-packages/omnimarket/nodes/"
    "node_hook_event_writer/handlers/handler_hook_event_writer.py"
)
HEAD = "8020dbdcc9a4961cdc5684d14c63b3bc024eef30"
DEV_COPY = b"# dev's writer, d25b2beef\n"
BRANCH_COPY = b"# the branch's writer, omnimarket#4312\n"
FEED_DEV_COPY = b"# dev's hook event writer\n"
FEED_BRANCH_COPY = b"# the branch's hook event writer\n"
NOW = datetime(2026, 10, 22, 11, 0, tzinfo=UTC)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _done(
    argv: list[str], code: int = 0, out: str = "", err: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, code, out, err)


class _Lane:
    """A fake Docker with one dev-lane runtime container built from an image."""

    def __init__(self, image: dict[str, bytes], project: str = PROJECT) -> None:
        self.project = project
        self.image = dict(image)
        self.generation = 0
        self.files: dict[str, bytes] = {}
        self.owner_mode: dict[str, str] = {}
        self.restarts: list[str] = []
        self.copies: list[str] = []
        self.container = ""
        self.running = True
        self.docker_down = False
        self.copy_is_ignored = False
        self.recreate()

    def recreate(self) -> None:
        """The dev refresh: a new container from the image, the layer is gone."""
        self.generation += 1
        self.container = f"{self.generation:064x}"
        self.files = dict(self.image)
        self.owner_mode = dict.fromkeys(self.image, "1000:1000 644")

    def install_by_hand(self, path: str, data: bytes) -> None:
        self.files[path] = data

    def __call__(
        self, argv: list[str], timeout: float = 0.0
    ) -> subprocess.CompletedProcess[str]:
        if self.docker_down:
            raise OSError("docker is not reachable")
        if argv[:2] == ["docker", "ps"]:
            wanted = {a for a in argv if a.startswith("label=")}
            if f"label=com.docker.compose.project={self.project}" not in wanted:
                return _done(argv)
            if f"label=com.docker.compose.service={SERVICE}" not in wanted:
                return _done(argv)
            return _done(argv, out=self.container + "\n" if self.running else "")
        if argv[:2] == ["docker", "exec"]:
            rest = argv[2:]
            if rest[:2] == ["-u", "0"]:
                rest = rest[2:]
            container, command = rest[0], rest[1:]
            assert container == self.container, "exec into a container that is gone"
            return self._exec(argv, command)
        if argv[:2] == ["docker", "cp"]:
            blob, _, target = argv[2], argv[3].partition(":")[0], argv[3]
            path = target.partition(":")[2]
            self.copies.append(path)
            if not self.copy_is_ignored:
                self.files[path] = Path(blob).read_bytes()
            return _done(argv)
        if argv[:2] == ["docker", "restart"]:
            self.restarts.append(argv[2])
            return _done(argv)
        if argv[:2] == ["docker", "inspect"]:
            return _done(argv, out="running|healthy\n")
        raise AssertionError(f"unexpected command {argv}")

    def _exec(
        self, argv: list[str], command: list[str]
    ) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["sha256sum", "--"]:
            path = command[2]
            if path not in self.files:
                return _done(
                    argv,
                    code=1,
                    err=f"sha256sum: {path}: No such file or directory",
                )
            return _done(argv, out=f"{_sha(self.files[path])}  {path}\n")
        if command[:1] == ["stat"]:
            return _done(argv, out=self.owner_mode.get(command[-1], "0:0 644") + "\n")
        if command[:1] == ["chown"]:
            path = command[-1]
            mode = self.owner_mode.get(path, "0:0 644").partition(" ")[2]
            self.owner_mode[path] = f"{command[1]} {mode}"
            return _done(argv)
        if command[:1] == ["chmod"]:
            path = command[-1]
            owner = self.owner_mode.get(path, "0:0 644").partition(" ")[0]
            self.owner_mode[path] = f"{owner} {command[1]}"
            return _done(argv)
        raise AssertionError(f"unexpected exec {command}")


def _lane() -> _Lane:
    return _Lane({WRITER: DEV_COPY, FEED: FEED_DEV_COPY})


def _store(tmp_path: Path) -> DevLaneBranchInstallStore:
    return DevLaneBranchInstallStore(tmp_path / "state")


def _source(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / "branch" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _declare(
    tmp_path: Path,
    store: DevLaneBranchInstallStore,
    *,
    install_id: str = "c28-writer-omnimarket-4312",
    pr: int = 4312,
    sources: dict[str, bytes] | None = None,
    hours: float = 48,
    restart: bool = True,
    replace: bool = False,
    lane: str = "dev",
) -> ModelDevLaneBranchInstall:
    files = sources or {WRITER: BRANCH_COPY}
    return register(
        store,
        lane=lane,
        install_id=install_id,
        repo="omnimarket",
        pr=pr,
        branch="jonah/omn-17427-c28-consumer-flow-writer",
        head=HEAD,
        ticket="OMN-17427",
        registered_by="c28-projection-writer-0720",
        services=[SERVICE],
        sources={
            path: _source(tmp_path, f"{pr}-{i}.py", data)
            for i, (path, data) in enumerate(sorted(files.items()))
        },
        expires_at=NOW + timedelta(hours=hours),
        restart=restart,
        replace=replace,
        now=NOW,
    )


def _handler(
    store: DevLaneBranchInstallStore,
    lane: _Lane,
    *,
    at: datetime = NOW,
    probe: Callable[..., str | None] | None = None,
) -> HandlerDevLaneBranchInstall:
    return HandlerDevLaneBranchInstall(
        store,
        PROJECT,
        run=lane,
        clock=lambda: at,
        sleep=lambda _s: None,
        merged_probe=probe,
    )


# --------------------------------------------------------------------------- #
# AC172: registration                                                         #
# --------------------------------------------------------------------------- #
def test_registration_keeps_the_bytes_by_hash_and_the_record_reads_back(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    install = _declare(tmp_path, store)

    records = store.load()
    assert [i.id for i in records.installs] == ["c28-writer-omnimarket-4312"]
    assert records.installs[0] == install
    assert records.installs[0].files[0].sha256 == _sha(BRANCH_COPY)
    assert store.blobs.path(_sha(BRANCH_COPY)).read_bytes() == BRANCH_COPY


@pytest.mark.parametrize("lane", ["prod", "stability", "lab", ""])
def test_registration_refuses_any_lane_but_dev_and_writes_nothing(
    tmp_path: Path, lane: str
) -> None:
    store = _store(tmp_path)
    with pytest.raises(DevLaneBranchInstallRefusedError, match="dev lane only"):
        _declare(tmp_path, store, lane=lane)
    assert not store.root.exists()


@pytest.mark.parametrize(
    "project",
    ["omnibase-infra-prod", "omnibase-infra-stability", "PROD", "STABILITY", ""],
)
def test_governed_compose_project_is_refused_before_any_docker_call(
    tmp_path: Path, project: str
) -> None:
    lane = _lane()
    with pytest.raises(DevLaneBranchInstallRefusedError, match="not the dev lane"):
        HandlerDevLaneBranchInstall(
            _store(tmp_path),
            project,
            run=lane,
        )
    assert lane.copies == [] and lane.restarts == []


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "/app",
        "/app/",
        "/app/../etc/passwd",
        "/app/x/./y.py",
        "/app//y.py",
        "/application/y.py",
        "/app/dir/",
        "relative/y.py",
    ],
)
def test_registration_refuses_a_path_outside_the_app_tree(
    tmp_path: Path, path: str
) -> None:
    store = _store(tmp_path)
    with pytest.raises(DevLaneBranchInstallRefusedError):
        _declare(tmp_path, store, sources={path: BRANCH_COPY})
    assert not store.records_path.exists()


def test_registration_refuses_an_expiry_in_the_past_or_beyond_seven_days(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    with pytest.raises(DevLaneBranchInstallRefusedError, match="not in the future"):
        _declare(tmp_path, store, hours=-1)
    with pytest.raises(DevLaneBranchInstallRefusedError, match="7 days"):
        _declare(tmp_path, store, hours=24 * 7 + 1)
    assert not store.records_path.exists()
    assert _declare(tmp_path, store, hours=24 * 7) is not None


def test_registration_refuses_a_source_whose_hash_is_not_the_one_shown(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    with pytest.raises(DevLaneBranchInstallRefusedError, match="expected"):
        register(
            store,
            lane="dev",
            install_id="c28-writer-omnimarket-4312",
            repo="omnimarket",
            pr=4312,
            branch="b",
            head=HEAD,
            ticket="OMN-17427",
            registered_by="lane",
            services=[SERVICE],
            sources={WRITER: _source(tmp_path, "w.py", BRANCH_COPY)},
            expires_at=NOW + timedelta(hours=2),
            expected_sha256={WRITER: _sha(b"another file")},
            now=NOW,
        )
    assert not store.records_path.exists()


def test_second_registration_of_one_id_needs_replace_and_keeps_the_first_in_history(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _declare(tmp_path, store)
    with pytest.raises(DevLaneBranchInstallRefusedError, match="already declared"):
        _declare(tmp_path, store, sources={WRITER: b"# a newer head\n"})
    assert store.load().installs[0].files[0].sha256 == _sha(BRANCH_COPY)

    _declare(tmp_path, store, sources={WRITER: b"# a newer head\n"}, replace=True)

    assert store.load().installs[0].files[0].sha256 == _sha(b"# a newer head\n")
    history = [
        json.loads(line) for line in store.released_path.read_text().splitlines()
    ]
    assert [h["release"]["reason"] for h in history] == ["replaced"]
    assert history[0]["install"]["files"][0]["sha256"] == _sha(BRANCH_COPY)


def test_a_corrupt_record_file_is_refused_never_read_as_no_installs(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.root.mkdir(parents=True)
    store.records_path.write_text("{not json")
    with pytest.raises(DevLaneBranchInstallRefusedError, match="not a valid record"):
        store.load()
    with pytest.raises(DevLaneBranchInstallRefusedError):
        _handler(store, _lane()).check()


# --------------------------------------------------------------------------- #
# AC173: the recreate that dropped the fix                                    #
# --------------------------------------------------------------------------- #
def test_replay_of_the_refresh_the_recreate_drops_the_fix_and_the_apply_puts_it_back(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store(tmp_path)
    lane = _lane()
    _declare(tmp_path, store)
    lane.install_by_hand(WRITER, BRANCH_COPY)
    handler = _handler(store, lane)
    assert handler.check().ok

    lane.recreate()  # the 11:18Z refresh: the writer is back to dev's copy
    assert lane.files[WRITER] == DEV_COPY
    with caplog.at_level(logging.INFO):
        before = handler.check()
    assert not before.ok

    receipt = handler.apply()

    assert receipt.ok
    assert lane.files[WRITER] == BRANCH_COPY
    assert _sha(lane.files[WRITER]) == store.load().installs[0].files[0].sha256
    readback = receipt.readbacks[0]
    assert readback.state is EnumDevLaneBranchInstallState.REAPPLIED
    assert readback.files[0].state is EnumDevLaneBranchInstallFileState.REAPPLIED
    assert lane.restarts == [lane.container], "restarted once, the new container"
    assert receipt.restarted == (lane.container,)
    assert handler.check().ok
    assert any("branch install REAPPLIED" in r.getMessage() for r in caplog.records)


def test_a_second_recreate_is_put_back_again_until_the_install_is_released(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    lane = _lane()
    _declare(tmp_path, store)
    handler = _handler(store, lane)
    for _ in range(3):
        lane.recreate()
        assert handler.apply().ok
        assert lane.files[WRITER] == BRANCH_COPY
    assert len(lane.restarts) == 3

    release(
        store,
        "c28-writer-omnimarket-4312",
        EnumDevLaneBranchInstallReleaseReason.OPERATOR,
        "merged",
    )
    lane.recreate()
    receipt = handler.apply()
    assert receipt.readbacks == ()
    assert lane.files[WRITER] == DEV_COPY, "a released install is not re-applied"


def test_apply_on_a_whole_lane_copies_nothing_and_restarts_nothing(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    lane = _lane()
    _declare(tmp_path, store)
    lane.install_by_hand(WRITER, BRANCH_COPY)

    receipt = _handler(store, lane).apply()

    assert receipt.ok
    assert receipt.readbacks[0].state is EnumDevLaneBranchInstallState.PRESENT
    assert lane.copies == [] and lane.restarts == []


def test_a_container_carrying_two_installs_is_restarted_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lane = _lane()
    _declare(tmp_path, store)
    _declare(
        tmp_path,
        store,
        install_id="hook-events-omnimarket-4304",
        pr=4304,
        sources={FEED: FEED_BRANCH_COPY},
    )
    lane.recreate()

    receipt = _handler(store, lane).apply()

    assert receipt.ok
    assert lane.files[WRITER] == BRANCH_COPY and lane.files[FEED] == FEED_BRANCH_COPY
    assert len(lane.restarts) == 1


def test_a_file_that_drifted_to_another_hash_is_replaced_keeping_owner_and_mode(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    lane = _lane()
    _declare(tmp_path, store)
    lane.install_by_hand(WRITER, b"# someone's hand edit\n")
    lane.owner_mode[WRITER] = "1000:1000 640"

    before = _handler(store, lane).check()
    assert (
        before.readbacks[0].files[0].state is EnumDevLaneBranchInstallFileState.DRIFTED
    )

    assert _handler(store, lane).apply().ok
    assert lane.files[WRITER] == BRANCH_COPY
    assert lane.owner_mode[WRITER] == "1000:1000 640"


def test_a_file_that_does_not_read_back_at_its_sha256_is_failed_not_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store(tmp_path)
    lane = _lane()
    _declare(tmp_path, store)
    lane.recreate()
    lane.copy_is_ignored = True

    with caplog.at_level(logging.INFO):
        receipt = _handler(store, lane).apply()

    assert not receipt.ok
    assert receipt.readbacks[0].state is EnumDevLaneBranchInstallState.FAILED
    assert (
        receipt.readbacks[0].files[0].state is EnumDevLaneBranchInstallFileState.FAILED
    )
    assert any(
        "BRANCH-INSTALL-MISSING" in r.getMessage() and r.levelno == logging.ERROR
        for r in caplog.records
    )


def test_a_corrupt_stored_copy_is_never_copied_into_a_container(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    lane = _lane()
    install = _declare(tmp_path, store)
    lane.recreate()
    store.blobs.path(install.files[0].sha256).write_bytes(b"truncated")

    receipt = _handler(store, lane).apply()

    assert not receipt.ok
    assert lane.copies == [] and lane.files[WRITER] == DEV_COPY
    assert "corrupt" in receipt.readbacks[0].files[0].detail


def test_an_install_that_asks_for_no_restart_is_copied_and_not_restarted(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    lane = _lane()
    _declare(tmp_path, store, restart=False)
    lane.recreate()

    assert _handler(store, lane).apply().ok
    assert lane.files[WRITER] == BRANCH_COPY
    assert lane.restarts == []


def test_a_service_with_no_running_container_is_failed_and_nothing_is_copied(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    lane = _lane()
    _declare(tmp_path, store)
    lane.running = False

    receipt = _handler(store, lane).apply()

    assert not receipt.ok
    assert (
        receipt.readbacks[0].files[0].state is EnumDevLaneBranchInstallFileState.FAILED
    )
    assert lane.copies == [] and lane.restarts == []


def test_a_project_with_other_containers_than_the_dev_lane_is_not_touched(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    prod = _Lane({WRITER: DEV_COPY}, project="omnibase-infra-prod")
    _declare(tmp_path, store)

    receipt = _handler(store, prod).apply()  # asks for project omnibase-infra

    assert not receipt.ok
    assert prod.copies == [] and prod.restarts == []
    assert prod.files[WRITER] == DEV_COPY


# --------------------------------------------------------------------------- #
# AC174: merged or expired                                                    #
# --------------------------------------------------------------------------- #
def test_an_expired_install_is_released_and_not_applied(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lane = _lane()
    _declare(tmp_path, store, hours=24)
    lane.recreate()

    receipt = _handler(store, lane, at=NOW + timedelta(hours=25)).apply()

    assert store.load().installs == ()
    assert [r.reason for r in receipt.released] == [
        EnumDevLaneBranchInstallReleaseReason.EXPIRED
    ]
    assert lane.copies == [] and lane.files[WRITER] == DEV_COPY
    assert not store.blobs.path(_sha(BRANCH_COPY)).exists(), (
        "its bytes are dropped with it"
    )


def test_release_history_keeps_the_whole_record_of_a_released_install(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    install = _declare(tmp_path, store)

    dropped = release(
        store,
        install.id,
        EnumDevLaneBranchInstallReleaseReason.OPERATOR,
        "the lane ended",
        now=NOW,
    )

    assert dropped is not None and dropped.detail == "the lane ended"
    row = json.loads(store.released_path.read_text().splitlines()[0])
    assert row["install"]["pr"] == 4312 and row["install"]["head"] == HEAD
    assert row["install"]["files"][0]["path"] == WRITER
    assert (
        release(store, install.id, EnumDevLaneBranchInstallReleaseReason.OPERATOR, "x")
        is None
    )


def _git(
    *, log: str = "", ancestor_of_dev: bool = False, in_running: bool = True
) -> Runner:
    def run(argv: list[str], timeout: float = 0.0) -> subprocess.CompletedProcess[str]:
        assert argv[0] == "git"
        if "log" in argv:
            return _done(argv, out=log)
        if "merge-base" in argv:
            if argv[-1] == "origin/dev":
                return _done(argv, code=0 if ancestor_of_dev else 1)
            return _done(argv, code=0 if in_running else 1)
        raise AssertionError(argv)

    return run


SQUASH = "a" * 40
SQUASH_LOG = f"{SQUASH}\x1ffix(OMN-17427): the writer folds windows (#4312)\n"


def _probe(**kwargs: object) -> Callable[..., str | None]:
    return git_merged_probe(
        lambda repo: Path("/clone") / repo,
        lambda repo: "c" * 40,
        run=_git(**kwargs),
    )


def test_a_merged_install_is_released_once_the_running_image_carries_the_merge(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    lane = _lane()
    _declare(tmp_path, store)
    lane.recreate()

    receipt = _handler(store, lane, probe=_probe(log=SQUASH_LOG)).apply()

    assert store.load().installs == ()
    assert [r.reason for r in receipt.released] == [
        EnumDevLaneBranchInstallReleaseReason.MERGED
    ]
    assert "omnimarket#4312 merged as aaaaaaaaaaaa" in receipt.released[0].detail
    assert lane.files[WRITER] == DEV_COPY, (
        "the merge is in the image, nothing to put back"
    )


def test_a_merge_the_running_image_does_not_yet_carry_keeps_the_install(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    lane = _lane()
    _declare(tmp_path, store)
    lane.recreate()

    receipt = _handler(
        store, lane, probe=_probe(log=SQUASH_LOG, in_running=False)
    ).apply()

    assert len(store.load().installs) == 1 and receipt.released == ()
    assert lane.files[WRITER] == BRANCH_COPY, "still applied until the image has it"


def test_a_branch_head_that_is_an_ancestor_of_dev_counts_as_merged(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _declare(tmp_path, store)

    released = _handler(
        store, _lane(), probe=_probe(log="", ancestor_of_dev=True)
    ).release_due()

    assert [r.reason for r in released] == [
        EnumDevLaneBranchInstallReleaseReason.MERGED
    ]


def test_an_unmerged_pr_and_an_unread_running_ref_keep_the_install(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _declare(tmp_path, store)

    unmerged = _probe(log=f"{SQUASH}\x1funrelated change (#4000)\n")
    unread = git_merged_probe(
        lambda repo: None,
        lambda repo: None,
        run=_git(log=SQUASH_LOG),
    )

    assert _handler(store, _lane(), probe=unmerged).release_due() == ()
    assert _handler(store, _lane(), probe=unread).release_due() == ()
    assert len(store.load().installs) == 1


# --------------------------------------------------------------------------- #
# AC175: the detector                                                         #
# --------------------------------------------------------------------------- #
def test_detected_install_missing_after_a_recreate_is_one_line_naming_both_hashes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store(tmp_path)
    lane = _lane()
    _declare(tmp_path, store)
    lane.recreate()

    with caplog.at_level(logging.INFO):
        receipt = _handler(store, lane).check()

    assert not receipt.ok
    lines = [r for r in caplog.records if "BRANCH-INSTALL-MISSING" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.ERROR
    text = lines[0].getMessage()
    for part in (
        "id=c28-writer-omnimarket-4312",
        "repo=omnimarket",
        "pr=#4312",
        f"file={WRITER}",
        f"expected_sha256={_sha(BRANCH_COPY)[:12]}",
        f"observed_sha256={_sha(DEV_COPY)[:12]}",
        "state=drifted",
        "friction_type=branch_install_missing",
    ):
        assert part in text
    assert lane.copies == [] and lane.restarts == [], "check is read only"


def test_an_unreadable_docker_is_missing_never_ok(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lane = _lane()
    _declare(tmp_path, store)
    lane.docker_down = True

    receipt = _handler(store, lane).check()

    assert not receipt.ok
    assert (
        receipt.readbacks[0].files[0].state
        is EnumDevLaneBranchInstallFileState.NO_CONTAINER
    )


def test_status_payload_lists_declared_installs_and_the_missing_ones_without_content(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    lane = _lane()
    _declare(tmp_path, store)
    lane.recreate()
    _handler(store, lane).check()

    body = status_payload(store, NOW)

    declared = body["declared"]
    assert isinstance(declared, list)
    assert [i["id"] for i in declared] == ["c28-writer-omnimarket-4312"]
    assert body["missing"] == ["c28-writer-omnimarket-4312"]
    assert BRANCH_COPY.decode() not in json.dumps(body)


def test_the_command_line_registers_applies_and_exits_not_whole_on_a_failed_readback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lane = _lane()
    source = _source(tmp_path, "w.py", BRANCH_COPY)
    state = str(tmp_path / "state")
    argv = [
        "register",
        "--state-dir",
        state,
        "--lane",
        "dev",
        "--id",
        "c28-writer-omnimarket-4312",
        "--repo",
        "omnimarket",
        "--pr",
        "4312",
        "--branch",
        "b",
        "--head",
        HEAD,
        "--ticket",
        "OMN-17427",
        "--registered-by",
        "lane",
        "--service",
        SERVICE,
        "--expires-in-hours",
        "24",
        "--file",
        f"{WRITER}={source}",
        "--expect-sha256",
        f"{WRITER}={_sha(BRANCH_COPY)}",
        "--compose-project",
        PROJECT,
    ]

    assert bi.main(argv, run=lane, sleep=lambda _s: None) == bi.EXIT_OK
    assert lane.files[WRITER] == BRANCH_COPY
    out = capsys.readouterr().out
    assert out.startswith("REGISTERED ") and "\nREADBACK " in out

    lane.recreate()
    lane.copy_is_ignored = True
    check = ["check", "--state-dir", state, "--compose-project", PROJECT]
    assert bi.main(check, run=lane) == bi.EXIT_NOT_WHOLE
    apply = ["apply", "--state-dir", state, "--compose-project", "omnibase-infra-prod"]
    assert bi.main(apply, run=lane) == bi.EXIT_REFUSED
