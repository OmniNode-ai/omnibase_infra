# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The individual signals composing runner readiness (OMN-15255, friction F-04).

Each member is an independent question about one runner. Readiness is the
conjunction of all of them -- no single member is sufficient, which is the
whole point: on 2026-07-27T16:40Z the GitHub registry reported 64/64 online
while 53/64 containers read docker-unhealthy. Either surface alone gives the
wrong answer.

Extending this enum does NOT require a model change: signals are carried as a
tuple of ``ModelRunnerReadinessSignal`` on the assessment. The two remaining
Recommended-4 signals (toolchain/image contract, tiny-governed-job probe) are
not implemented yet and are deliberately absent rather than stubbed PASS.
"""

from __future__ import annotations

from enum import StrEnum


class EnumRunnerReadinessSignal(StrEnum):
    """One independently-probed readiness question."""

    GITHUB_REGISTRATION = "github_registration"
    """GitHub org registry reports this runner ``online``. Necessary: work is
    dispatched by GitHub, so an offline runner cannot receive it regardless of
    how healthy the container looks locally."""

    DOCKER_HEALTH = "docker_health"
    """The container's Docker health status is ``healthy`` (or ``none`` where
    the image declares no healthcheck). Not an input to the legacy precedence
    classifier at all -- the blind spot F-04 names."""

    DIAG_HEARTBEAT = "diag_heartbeat"
    """Newest ``_diag/*.log`` write is within the idle-cadence threshold.
    OMN-15233: the threshold must bracket a full Runner.Listener token-refresh
    cycle (measured ~50-53 min), which the retired 900s default did not."""

    LISTENER_TOPOLOGY = "listener_topology"
    """Exactly one ``Runner.Listener`` process, zero PPID-1 orphans. A
    double-listener or an orphan reparented to init produces
    ``TaskAgentSessionConflictException`` on the replacement, which the
    registry still reports as ``online``."""

    CONTAINER_STABILITY = "container_stability"
    """Docker ``RestartCount`` is at or under the crash-loop threshold. Folded
    in as a readiness signal (OMN-15255) so ``RESTART_RUNNER`` has exactly one
    producer instead of a state-keyed branch plus a readiness rule."""

    DISK_CAPACITY = "disk_capacity"
    """Runner-host disk usage is under the ceiling. Deliberately NOT
    bounce-remediable: recreating a container frees no host disk, so a disk
    failure quarantines the runner and recommends no restart."""


__all__ = ["EnumRunnerReadinessSignal"]
