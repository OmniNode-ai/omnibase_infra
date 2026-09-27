# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Defer a deploy while the host's model servers need the machine (OMN-19507).

WHY THIS EXISTS
---------------
Operator RULING, omni_home ledger 2026-09-25T12:40:36Z, item (b): "add a deploy
lane on .200 with a load gate". The .200 host (Stickybeatz-Studio, 24 cores,
192 GiB unified memory) is also where the orchestration sessions run and where
the gpt-oss-120b llama-server on port 8130 serves the local review model the
same ruling adopts (item a). A lane rebuild is a workspace build plus a compose
recreate inside a Docker VM capped at 31 GiB and 24 vCPUs, so an unconditional
deploy can take the memory the model's weights live in. On Apple silicon GPU
memory IS system memory: a model whose pages are squeezed out answers from swap
or not at all.

So the instance declares a gate in ``config/deploy_lane_routing.yaml``
(``load_gate:``) and its agent does not ACCEPT a command while the gate is shut.
Deferral is not a rejection: nothing is committed and nothing is published, the
record stays on the control topic, and the agent takes it (or a newer one it
coalesces to) once a re-check opens the gate.

THE DECISION (``decide``), one reading, two rules
-------------------------------------------------
* CPU: ``load1 / cpu_count`` above ``max_load1_per_cpu`` defers.
* Memory, with the GPU folded in: the headroom a build would leave,

      headroom = free - build_reserve - max(0, model_reserve - gpu_in_use)

  below ``min_headroom_gib`` defers. ``build_reserve`` is what the Docker VM can
  still grow by during a rebuild; ``model_reserve`` is the model server's own
  working set when it is serving. A model whose weights are resident is already
  counted in ``free``; one that has been paged out takes its working set back
  the moment a request arrives, so the unmapped part is charged up front.

Every threshold is data, and the numbers for .200 are measured, not invented:
the PR that declares them states each baseline and where it was read.

AN UNREADABLE INPUT DEFERS. This is the opposite direction from
``host_conditions`` (a kill ceiling widens on a bad read, because a ceiling is
not an authorization). A gate is one: a deploy that cannot tell whether it would
starve the model server does not start. The deferral names the input it could
not read, so a gate that is shut for that reason is visible, not silent.

THE PROBE NEVER RAISES. Each reader is a keyword seam; a reader that throws
records ``None``, which the decision turns into a named deferral.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from deploy_agent.routing import ROUTING_TABLE_RELPATH, RoutingTableError

logger = logging.getLogger(__name__)

#: The key an instance's gate sits under in the routing table.
LOAD_GATE_KEY: Final = "load_gate"

#: How long one host reader may take before it is abandoned as unreadable.
READER_TIMEOUT_SECONDS: Final = 10

GIB: Final = 1024**3

_FREE_PERCENT_RE: Final = re.compile(
    r"System-wide memory free percentage:\s*([0-9]+(?:\.[0-9]+)?)%"
)
_GPU_IN_USE_RE: Final = re.compile(r'"In use system memory"=([0-9]+)')


class ModelLoadGateThresholds(BaseModel):
    """One instance's ``load_gate:`` block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_load1_per_cpu: float = Field(gt=0)
    build_reserve_gib: float = Field(ge=0)
    model_reserve_gib: float = Field(ge=0)
    min_headroom_gib: float = Field(ge=0)
    recheck_seconds: int = Field(gt=0)


class ModelHostLoadReading(BaseModel):
    """One reading of the host. Every ``None`` is a reader that failed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    load1: float | None
    cpu_count: int | None
    memory_total_gib: float | None
    memory_free_percent: float | None
    gpu_in_use_gib: float | None


class EnumLoadGateVerdict(StrEnum):
    OPEN = "open"
    DEFER = "defer"


class ModelLoadGateDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: EnumLoadGateVerdict
    reasons: tuple[str, ...]
    reading: ModelHostLoadReading
    headroom_gib: float | None

    def describe(self) -> str:
        """One log-ready line: the verdict, every reading, and why."""
        r = self.reading
        load = (
            f"load1 {r.load1:.2f} over {r.cpu_count} cpu"
            if r.load1 is not None and r.cpu_count is not None
            else "load1 unreadable"
        )
        free = (
            f"free {r.memory_free_percent:.0f}% of {r.memory_total_gib:.0f} GiB"
            if r.memory_free_percent is not None and r.memory_total_gib is not None
            else "free memory unreadable"
        )
        gpu = (
            f"gpu in use {r.gpu_in_use_gib:.1f} GiB"
            if r.gpu_in_use_gib is not None
            else "gpu in use unreadable"
        )
        headroom = (
            f"headroom {self.headroom_gib:.1f} GiB"
            if self.headroom_gib is not None
            else "headroom unknown"
        )
        why = "; ".join(self.reasons) if self.reasons else "within every threshold"
        return f"{self.verdict.value}: {load}, {free}, {gpu}, {headroom} ({why})"


def decide(
    thresholds: ModelLoadGateThresholds, reading: ModelHostLoadReading
) -> ModelLoadGateDecision:
    """The gate's verdict for one reading. Pure."""
    reasons: list[str] = []
    for field in ModelHostLoadReading.model_fields:
        if getattr(reading, field) is None:
            reasons.append(f"{field} unreadable")

    if reading.load1 is not None and reading.cpu_count:
        saturation = reading.load1 / reading.cpu_count
        if saturation > thresholds.max_load1_per_cpu:
            reasons.append(
                f"load1 {reading.load1:.2f} over {reading.cpu_count} cpu is "
                f"{saturation:.2f} per cpu, above {thresholds.max_load1_per_cpu:.2f}"
            )

    headroom: float | None = None
    if (
        reading.memory_total_gib is not None
        and reading.memory_free_percent is not None
        and reading.gpu_in_use_gib is not None
    ):
        free_gib = reading.memory_total_gib * reading.memory_free_percent / 100
        unmapped = max(0.0, thresholds.model_reserve_gib - reading.gpu_in_use_gib)
        headroom = free_gib - thresholds.build_reserve_gib - unmapped
        if headroom < thresholds.min_headroom_gib:
            reasons.append(
                f"headroom {headroom:.1f} GiB (free {free_gib:.1f} less build "
                f"reserve {thresholds.build_reserve_gib:.0f} less unmapped model "
                f"reserve {unmapped:.1f}) is below {thresholds.min_headroom_gib:.0f}"
            )

    return ModelLoadGateDecision(
        verdict=EnumLoadGateVerdict.DEFER if reasons else EnumLoadGateVerdict.OPEN,
        reasons=tuple(reasons),
        reading=reading,
        headroom_gib=headroom,
    )


# --- the host readers ------------------------------------------------------------


def parse_memory_free_percent(text: str) -> float | None:
    """macOS ``memory_pressure``: the system-wide free percentage."""
    match = _FREE_PERCENT_RE.search(text)
    return float(match.group(1)) if match else None


def parse_gpu_in_use_bytes(text: str) -> int | None:
    """macOS ``ioreg -c IOAccelerator``: the GPU's in-use system memory."""
    match = _GPU_IN_USE_RE.search(text)
    return int(match.group(1)) if match else None


def _run(argv: list[str]) -> str:
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        timeout=READER_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        msg = f"{argv[0]} exited {result.returncode}: {result.stderr[:200]}"
        raise RuntimeError(msg)
    return result.stdout


def _default_loadavg() -> tuple[float, float, float]:
    return os.getloadavg()


def _default_cpu_count() -> int | None:
    return os.cpu_count()


def _default_memory_total_bytes() -> int:
    return int(_run(["sysctl", "-n", "hw.memsize"]).strip())


def _default_memory_free_percent() -> float | None:
    return parse_memory_free_percent(_run(["memory_pressure"]))


def _default_gpu_in_use_bytes() -> int | None:
    return parse_gpu_in_use_bytes(
        _run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
    )


def _safe[T](reader: Callable[[], T], name: str) -> T | None:
    try:
        return reader()
    except Exception as exc:  # noqa: BLE001 - an unread input defers, by name
        logger.warning(
            "load gate: %s unreadable (%s: %s)", name, type(exc).__name__, exc
        )
        return None


def probe_host_load(
    *,
    read_loadavg: Callable[[], tuple[float, float, float]] = _default_loadavg,
    read_cpu_count: Callable[[], int | None] = _default_cpu_count,
    read_memory_total_bytes: Callable[[], int] = _default_memory_total_bytes,
    read_memory_free_percent: Callable[[], float | None] = _default_memory_free_percent,
    read_gpu_in_use_bytes: Callable[[], int | None] = _default_gpu_in_use_bytes,
) -> ModelHostLoadReading:
    """Read the host once. Never raises."""
    loadavg = _safe(read_loadavg, "load1")
    total = _safe(read_memory_total_bytes, "memory_total_gib")
    gpu = _safe(read_gpu_in_use_bytes, "gpu_in_use_gib")
    return ModelHostLoadReading(
        load1=loadavg[0] if loadavg else None,
        cpu_count=_safe(read_cpu_count, "cpu_count"),
        memory_total_gib=total / GIB if total else None,
        memory_free_percent=_safe(read_memory_free_percent, "memory_free_percent"),
        gpu_in_use_gib=gpu / GIB if gpu is not None else None,
    )


class LoadGate:
    """An instance's gate: its thresholds, a reader, and a re-check clock."""

    def __init__(
        self,
        thresholds: ModelLoadGateThresholds,
        *,
        read: Callable[[], ModelHostLoadReading] = probe_host_load,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.thresholds = thresholds
        self._read = read
        self._clock = clock
        self._last_check: float | None = None

    def check(self) -> ModelLoadGateDecision:
        """A fresh reading and its verdict. Stamps the re-check clock."""
        self._last_check = self._clock()
        return decide(self.thresholds, self._read())

    def recheck_due(self) -> bool:
        """Whether a deferral may read the host again yet."""
        return (
            self._last_check is None
            or self._clock() - self._last_check >= self.thresholds.recheck_seconds
        )


# --- the table -------------------------------------------------------------------


def parse_load_gates(text: str) -> dict[str, ModelLoadGateThresholds]:
    """Every instance's ``load_gate:`` block, keyed by instance name.

    An instance without a block has no gate. A block that does not validate
    refuses, and the agent reading its own table then refuses to start, as it
    does for a malformed routing table.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RoutingTableError(f"routing table is not YAML: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("instances"), dict):
        raise RoutingTableError("routing table declares no instances")
    gates: dict[str, ModelLoadGateThresholds] = {}
    for name, spec in raw["instances"].items():
        if not isinstance(spec, dict) or LOAD_GATE_KEY not in spec:
            continue
        block = spec[LOAD_GATE_KEY]
        if not isinstance(block, dict):
            raise RoutingTableError(
                f"instance {name!r} {LOAD_GATE_KEY}: must be a mapping"
            )
        try:
            gates[str(name)] = ModelLoadGateThresholds.model_validate(block)
        except ValidationError as exc:
            raise RoutingTableError(
                f"instance {name!r} {LOAD_GATE_KEY}: block is invalid: {exc}"
            ) from exc
    return gates


def load_gate_for_instance(
    repo_root: str | Path, instance: str
) -> ModelLoadGateThresholds | None:
    """The gate ``instance`` declares in ``repo_root``'s committed table."""
    path = Path(repo_root) / ROUTING_TABLE_RELPATH
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RoutingTableError(f"routing table unreadable at {path}: {exc}") from exc
    return parse_load_gates(text).get(instance)
