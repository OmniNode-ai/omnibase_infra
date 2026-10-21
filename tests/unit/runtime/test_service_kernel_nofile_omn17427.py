# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime kernel raises its open-file soft limit before boot (OMN-17427)."""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Coroutine

import pytest

from omnibase_infra.runtime import service_kernel
from omnibase_infra.runtime.service_kernel import raise_nofile_soft_limit

pytestmark = pytest.mark.unit

_INFINITY_CAP = 1_048_576


class FakeResource:
    """Stateful stand-in for the resource module that can refuse a write."""

    RLIMIT_NOFILE = 7
    RLIM_INFINITY = -1

    def __init__(
        self, soft: int, hard: int, error: ValueError | OSError | None = None
    ) -> None:
        self.limits = (soft, hard)
        self.error = error
        self.calls: list[tuple[int, int]] = []

    def getrlimit(self, which: int, /) -> tuple[int, int]:
        assert which == self.RLIMIT_NOFILE
        return self.limits

    def setrlimit(self, which: int, limits: tuple[int, int], /) -> None:
        assert which == self.RLIMIT_NOFILE
        self.calls.append(limits)
        if self.error is not None:
            raise self.error
        self.limits = limits


def _install(monkeypatch: pytest.MonkeyPatch, fake: FakeResource | None) -> None:
    # A None entry in sys.modules makes `import resource` raise ImportError.
    monkeypatch.setitem(sys.modules, "resource", fake)


def test_raises_soft_to_hard(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeResource(1024, 524288)
    _install(monkeypatch, fake)

    with caplog.at_level(logging.INFO, logger=service_kernel.__name__):
        raise_nofile_soft_limit()

    assert fake.calls == [(524288, 524288)]
    assert fake.limits == (524288, 524288)
    assert "raised from 1024 to 524288 (hard 524288)" in caplog.text


@pytest.mark.parametrize(
    ("soft", "hard"),
    [
        (524288, 524288),
        (FakeResource.RLIM_INFINITY, FakeResource.RLIM_INFINITY),
        (_INFINITY_CAP + 1, FakeResource.RLIM_INFINITY),
    ],
)
def test_leaves_soft_at_or_above_target(
    monkeypatch: pytest.MonkeyPatch, soft: int, hard: int
) -> None:
    fake = FakeResource(soft, hard)
    _install(monkeypatch, fake)

    raise_nofile_soft_limit()

    assert fake.calls == []
    assert fake.limits == (soft, hard)


def test_caps_unlimited_hard_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeResource(1024, FakeResource.RLIM_INFINITY)
    _install(monkeypatch, fake)

    raise_nofile_soft_limit()

    assert fake.calls == [(_INFINITY_CAP, FakeResource.RLIM_INFINITY)]


@pytest.mark.parametrize("error", [ValueError("refused"), OSError("refused")])
def test_refused_raise_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: ValueError | OSError,
) -> None:
    fake = FakeResource(1024, 524288, error)
    _install(monkeypatch, fake)

    with caplog.at_level(logging.WARNING, logger=service_kernel.__name__):
        raise_nofile_soft_limit()

    assert fake.limits == (1024, 524288)
    assert f"left at 1024 (hard 524288): {type(error).__name__}: refused" in (
        caplog.text
    )


def test_platform_without_resource_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, None)

    raise_nofile_soft_limit()


@pytest.mark.skipif(sys.platform == "win32", reason="resource is Unix-only")
def test_subprocess_raises_real_soft_limit_from_256() -> None:
    code = """
import resource

_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
before, _ = resource.getrlimit(resource.RLIMIT_NOFILE)

from omnibase_infra.runtime.service_kernel import raise_nofile_soft_limit

raise_nofile_soft_limit()
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
print(before, soft, hard, resource.RLIM_INFINITY)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
    )
    before, soft, hard, infinity = map(int, completed.stdout.split()[-4:])

    assert before == 256
    assert soft == (_INFINITY_CAP if hard == infinity else hard)


def test_kernel_raises_limit_before_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    def fake_run(coro: Coroutine[object, object, int]) -> int:
        order.append("run")
        coro.close()
        return 0

    monkeypatch.setattr(service_kernel, "configure_logging", lambda: None)
    monkeypatch.setattr(
        service_kernel, "raise_nofile_soft_limit", lambda: order.append("nofile")
    )
    monkeypatch.setattr(service_kernel.asyncio, "run", fake_run)
    monkeypatch.setattr(
        "omnibase_infra.runtime.tracing.configure_tracing", lambda: None
    )

    with pytest.raises(SystemExit) as exc_info:
        service_kernel.main()

    assert exc_info.value.code == 0
    assert order == ["nofile", "run"]
