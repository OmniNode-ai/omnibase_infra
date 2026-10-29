# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21426 AC1 and AC2 against a real etcd and a real git remote.

Two lease orchestrators, one per simulated runtime, share the single-member
test etcd of OMN-21421 and OMN-21424. A bare git repository on disk stands in
for GitHub: a push that reaches it moves its branch ref. The model is
``formal/generic_lease/GenericLease.tla`` (OMN-21418): a fix worker's push is
an effect carrying its fence, and ``NoEffectWithStaleFence`` holds when the
fences the remote accepted never go down.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.lease import (
    HandlerLeaseAcquire,
    HandlerLeaseRelease,
    HandlerLeaseStore,
)
from omnibase_infra.handlers.lease.handler_fenced_push import (
    GitCliPushTransport,
    HandlerFencedPush,
)
from omnibase_infra.handlers.lease.handler_landing_lease import (
    LANDING_LEASE_KINDS,
    EnumLandingLeaseRow,
    landing_lease_request,
)
from omnibase_infra.handlers.placement import HandlerPlacementEtcd
from omnibase_infra.models.lease import (
    ModelFencedPushRequest,
    ModelLeaseReleaseRequest,
    ModelLeaseReply,
)

from .conftest import EtcdProcess

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

PR = "omnimarket#4151"
BRANCH = "omn-4151-fix"


@pytest.fixture
async def two_runtimes(
    etcd: EtcdProcess,
) -> AsyncIterator[tuple[HandlerPlacementEtcd, HandlerPlacementEtcd]]:
    adapters = (
        HandlerPlacementEtcd(etcd.endpoint, httpx.AsyncClient(timeout=10.0)),
        HandlerPlacementEtcd(etcd.endpoint, httpx.AsyncClient(timeout=10.0)),
    )
    try:
        yield adapters
    finally:
        for adapter in adapters:
            await adapter.aclose()


def _acquire(adapter: HandlerPlacementEtcd) -> HandlerLeaseAcquire:
    return HandlerLeaseAcquire(HandlerLeaseStore(adapter, LANDING_LEASE_KINDS))


def _release(adapter: HandlerPlacementEtcd) -> HandlerLeaseRelease:
    return HandlerLeaseRelease(HandlerLeaseStore(adapter, LANDING_LEASE_KINDS))


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(),
    ).stdout.strip()


class Remote:
    """A bare repository standing in for GitHub, and one clone per worker."""

    def __init__(self, root: Path) -> None:
        self.bare = root / "github.git"
        _git(root, "init", "--bare", "-b", "main", str(self.bare))
        seed = self._clone(root / "seed")
        _git(seed, "commit", "--allow-empty", "-m", "base")
        _git(seed, "push", "origin", f"HEAD:refs/heads/{BRANCH}")

    def _clone(self, path: Path) -> Path:
        _git(path.parent, "clone", "-q", str(self.bare), str(path))
        _git(path, "config", "user.email", "worker@example.invalid")
        _git(path, "config", "user.name", "worker")
        return path

    def clone(self, path: Path) -> Path:
        self._clone(path)
        _git(path, "checkout", "-q", "-B", BRANCH, f"origin/{BRANCH}")
        return path

    def head(self) -> str:
        return _git(self.bare, "rev-parse", f"refs/heads/{BRANCH}")


def _commit(clone: Path, message: str) -> str:
    _git(clone, "commit", "--allow-empty", "-m", message)
    return _git(clone, "rev-parse", "HEAD")


def _push_request(
    grant: ModelLeaseReply, clone: Path, expected: str
) -> ModelFencedPushRequest:
    return ModelFencedPushRequest(
        kind=grant.kind,
        resource=grant.resource,
        fence=grant.fence,
        worktree=str(clone),
        remote="origin",
        branch=BRANCH,
        expected_remote_sha=expected,
    )


async def test_ac1_every_landing_row_is_granted_by_the_lease_orchestrator(
    two_runtimes: tuple[HandlerPlacementEtcd, HandlerPlacementEtcd],
) -> None:
    a, b = two_runtimes
    subjects = {
        EnumLandingLeaseRow.A1_FIX_WORKER: PR,
        EnumLandingLeaseRow.A2_CAUSE: "ruff-red-1",
        EnumLandingLeaseRow.A3_LAB_PROVER: "omnimarket",
    }
    for row, subject in subjects.items():
        replies = await asyncio.gather(
            *(
                _acquire(a if i % 2 else b).handle(
                    landing_lease_request(
                        row, subject, holder=f"worker-{i}", request_id=uuid4()
                    )
                )
                for i in range(20)
            )
        )
        outcomes = [reply.outcome for reply in replies]
        assert outcomes.count("lease-granted") == 1, (row, outcomes)
        assert outcomes.count("lease-refused") == 19
    owner = await _acquire(a).handle(
        landing_lease_request(
            EnumLandingLeaseRow.D5_CI_RED_OWNER,
            "ruff-red-1",
            holder="ci-red-owner",
            request_id=uuid4(),
        )
    )
    assert owner.outcome == "lease-refused", "D5 and A2 must share one owner per cause"


async def test_ac2_revoked_fix_worker_push_is_refused_lease_fence_stale(
    two_runtimes: tuple[HandlerPlacementEtcd, HandlerPlacementEtcd],
    tmp_path: Path,
) -> None:
    a, b = two_runtimes
    remote = Remote(tmp_path)
    base = remote.head()
    old_clone = remote.clone(tmp_path / "old-worker")
    new_clone = remote.clone(tmp_path / "new-worker")
    old = await _acquire(a).handle(
        landing_lease_request(
            EnumLandingLeaseRow.A1_FIX_WORKER, PR, holder="old", request_id=uuid4()
        )
    )
    assert old.outcome == "lease-granted" and old.lease_id is not None
    revoked = await _release(a).handle(
        ModelLeaseReleaseRequest(
            kind=old.kind, resource=old.resource, lease_id=old.lease_id
        )
    )
    assert revoked.outcome == "lease-released"
    new = await _acquire(b).handle(
        landing_lease_request(
            EnumLandingLeaseRow.A1_FIX_WORKER, PR, holder="new", request_id=uuid4()
        )
    )
    assert new.outcome == "lease-granted" and new.fence > old.fence

    stale_commit = _commit(old_clone, "revoked worker's fix")
    pusher_a = HandlerFencedPush(a, GitCliPushTransport().push)
    refused = await pusher_a.handle(_push_request(old, old_clone, base))
    assert refused.outcome == "refused"
    assert refused.reason == "LEASE_FENCE_STALE"
    assert remote.head() == base, "the revoked worker's push reached the remote"

    new_commit = _commit(new_clone, "live worker's fix")
    pusher_b = HandlerFencedPush(b, GitCliPushTransport().push)
    pushed = await pusher_b.handle(_push_request(new, new_clone, base))
    assert pushed.outcome == "pushed", pushed
    assert remote.head() == new_commit != stale_commit

    again = await pusher_a.handle(_push_request(old, old_clone, new_commit))
    assert (again.outcome, again.reason) == ("refused", "LEASE_FENCE_STALE")
    assert remote.head() == new_commit


async def test_no_effect_with_stale_fence_over_interleaved_pushes(
    two_runtimes: tuple[HandlerPlacementEtcd, HandlerPlacementEtcd],
    tmp_path: Path,
) -> None:
    """GenericLease ``NoEffectWithStaleFence``: accepted fences never go down."""
    a, b = two_runtimes
    remote = Remote(tmp_path)
    applied: list[int] = []
    clones = [remote.clone(tmp_path / f"w{i}") for i in range(3)]
    grants: list[ModelLeaseReply] = []
    for i, clone in enumerate(clones):
        adapter = a if i % 2 else b
        if grants:
            prior = grants[-1]
            assert prior.lease_id is not None
            await _release(adapter).handle(
                ModelLeaseReleaseRequest(
                    kind=prior.kind, resource=prior.resource, lease_id=prior.lease_id
                )
            )
        grant = await _acquire(adapter).handle(
            landing_lease_request(
                EnumLandingLeaseRow.A1_FIX_WORKER,
                PR,
                holder=f"w{i}",
                request_id=uuid4(),
            )
        )
        assert grant.outcome == "lease-granted"
        grants.append(grant)
        for j, (earlier, earlier_clone) in enumerate(zip(grants, clones, strict=False)):
            _commit(earlier_clone, f"w{j} after grant {i}")
            pusher = HandlerFencedPush(adapter, GitCliPushTransport().push)
            result = await pusher.handle(
                _push_request(earlier, earlier_clone, remote.head())
            )
            if result.outcome == "pushed":
                applied.append(earlier.fence)
            else:
                assert j < i and result.reason == "LEASE_FENCE_STALE", result
    assert applied == sorted(applied) and len(applied) == 3


async def test_a_released_grant_with_no_successor_is_refused_lease_expired(
    two_runtimes: tuple[HandlerPlacementEtcd, HandlerPlacementEtcd],
    tmp_path: Path,
) -> None:
    a, _ = two_runtimes
    remote = Remote(tmp_path)
    base = remote.head()
    clone = remote.clone(tmp_path / "worker")
    grant = await _acquire(a).handle(
        landing_lease_request(
            EnumLandingLeaseRow.A1_FIX_WORKER, PR, holder="w", request_id=uuid4()
        )
    )
    assert grant.lease_id is not None
    await _release(a).handle(
        ModelLeaseReleaseRequest(
            kind=grant.kind, resource=grant.resource, lease_id=grant.lease_id
        )
    )
    _commit(clone, "after release")
    result = await HandlerFencedPush(a, GitCliPushTransport().push).handle(
        _push_request(grant, clone, base)
    )
    assert (result.outcome, result.reason) == ("refused", "LEASE_EXPIRED")
    assert remote.head() == base


async def test_an_unreachable_store_refuses_the_push_lease_unreadable(
    etcd: EtcdProcess, tmp_path: Path
) -> None:
    remote = Remote(tmp_path)
    base = remote.head()
    clone = remote.clone(tmp_path / "worker")
    adapter = HandlerPlacementEtcd(etcd.endpoint, httpx.AsyncClient(timeout=2.0))
    try:
        grant = await _acquire(adapter).handle(
            landing_lease_request(
                EnumLandingLeaseRow.A1_FIX_WORKER, PR, holder="w", request_id=uuid4()
            )
        )
        etcd.stop()
        _commit(clone, "while etcd is down")
        result = await HandlerFencedPush(adapter, GitCliPushTransport().push).handle(
            _push_request(grant, clone, base)
        )
    finally:
        await adapter.aclose()
    assert (result.outcome, result.reason) == ("refused", "LEASE_UNREADABLE")
    assert remote.head() == base


async def test_a_live_fence_still_cannot_overwrite_a_moved_remote(
    two_runtimes: tuple[HandlerPlacementEtcd, HandlerPlacementEtcd],
    tmp_path: Path,
) -> None:
    """The residual window of plan 5.4: ``--force-with-lease`` on the expected sha."""
    a, _ = two_runtimes
    remote = Remote(tmp_path)
    base = remote.head()
    clone = remote.clone(tmp_path / "worker")
    other = remote.clone(tmp_path / "other")
    moved = _commit(other, "someone else")
    _git(other, "push", "origin", f"HEAD:refs/heads/{BRANCH}")
    grant = await _acquire(a).handle(
        landing_lease_request(
            EnumLandingLeaseRow.A1_FIX_WORKER, PR, holder="w", request_id=uuid4()
        )
    )
    _commit(clone, "on the old base")
    result = await HandlerFencedPush(a, GitCliPushTransport().push).handle(
        _push_request(grant, clone, base)
    )
    assert (result.outcome, result.reason) == ("refused", "PUSH_REJECTED")
    assert remote.head() == moved
