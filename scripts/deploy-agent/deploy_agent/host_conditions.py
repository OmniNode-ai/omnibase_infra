# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the machine an image build is about to run on (OMN-18615).

WHY THIS EXISTS
---------------

OMN-18072 replaced a flat ``PHASE_TIMEOUTS[Phase.RUNTIME] = 300`` with a
ceiling derived from the build MODEL: the Dockerfile's work-step count and the
profile's buildable-service count. That mechanism works and is not in question
here. What it cannot express is that **both of its terms are properties of the
repository**. Neither is a property of the machine, so the number cannot move
when the machine does.

MEASURED on the ``.201`` dev lane, 2026-09-17. The same rebuild
(``omnimarket#2622`` squash ``387d5fe4cbfa5d04baf0872880d70fa83268e35e``) was
killed twice in one hour, at the byte-identical ceiling, across a 37-point
spread in load average:

    b046ce97  16:19:40Z -> 16:37:41Z  1081s  load1 ~97   ceiling 1080s
    499ab623  17:23:32Z -> 17:41:35Z  1083s  load1 ~60   ceiling 1080s

The same tree built in about 6m44s at 11:43Z that morning, warm, under a 1240s
ceiling. Load alone does not explain the pair -- the second kill ran at roughly
60 rather than 100 and still landed within two seconds of the first -- so this
module reads the host properties the model cannot express:

* **contention**, as ``load1 / cpu_count``: how much of the machine this build
  is actually going to get;
* **BuildKit cache state**: ``docker builder prune --force --filter until=3h``
  runs on this host and concurrent CI image builds evict the shared cache, so a
  cold-cache build and a warm-cache build of an identical tree are otherwise
  bounded by the same number;
* **I/O pressure**, as Linux PSI's ``some`` percentage: time work is stalled
  waiting for I/O, even when the CPU saturation term reads idle.

MEASURED on the ``.202`` lab host, instance ``dev-202``, 2026-10-02. Four
runtime builds in a row died with ``runtime_image_build_budget_exhausted`` at
1095s. One completed 51/61 work steps and exported 0 images: 21.5s/step
observed against 15s/step assumed; another completed 42/61 (~26s/step).
The host term read ``load1 14.00 over 32 cpu (saturation 0.44, x1.00)``,
but ``/proc/pressure/io`` read:

    some avg10=48.90 avg60=40.95 avg300=50.99
    full avg10=45.96 avg60=39.13 avg300=48.83

NVMe utilisation was 88-93 percent. The load term missed these I/O stalls,
so the ceiling never widened. Reading the larger sustained ``some`` average
(60s or 300s) charges that 50.99 percent at x1.82, opening 1095s to ~1993s
without changing the hard upper bound.

THE DIRECTION OF EVERY DEGRADED READING IS OUTWARD, AND THAT IS NOT A
FAIL-OPEN. A ceiling is not an authorization gate; it is a kill switch for a
hung build. Over-estimating it only delays a kill, while under-estimating it
kills a healthy build -- which is the failure OMN-18072 existed to remove and
the failure this ticket is. So an unreadable ``loadavg``, an unreadable cpu
count, an unreachable docker and unreadable I/O PSI all resolve to ``UNKNOWN``
and WIDEN. PSI absent from the platform is ``NOT_EXPOSED``, charged x1.00:
that is a platform fact, not a failed reading. The safety property that does
fail closed is the one in ``build_budget``:
``HARD_UPPER_BOUND_SECONDS``, which no combination of terms can exceed, and
which is what keeps "adapts" from meaning "unbounded" (OMN-18615 AC5).

THIS MODULE NEVER RAISES. ``probe_host_conditions`` catches everything its
readers can throw, because a deploy agent that could not derive a ceiling at
all would refuse every rebuild -- a strictly worse outcome than a wide one.
Its readers are keyword seams so a test can drive an idle host and a contended
one without touching the real ``os`` call, PSI file or docker daemon.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class EnumBuildCacheState(StrEnum):
    """How much BuildKit cache this build can expect to reuse.

    ``UNKNOWN`` is a third value rather than a default of ``COLD`` because the
    two are different facts, and the kill message has to be able to say which
    one it had. They happen to widen the ceiling by the same factor today; that
    is a calibration, not an identity, and collapsing them would make a future
    recalibration silently impossible.
    """

    WARM = "warm"
    COLD = "cold"
    UNKNOWN = "unknown"


class EnumIoPressureState(StrEnum):
    """A measured stall percentage, an absent PSI interface, or a failed read."""

    READ = "read"
    NOT_EXPOSED = "not_exposed"
    UNKNOWN = "unknown"


#: Above this saturation ratio (``load1 / cpu_count``) the build starts
#: competing for CPU it was implicitly assumed to have. At or below it the
#: ceiling is the model's own number, unchanged.
CONTENTION_SATURATION_THRESHOLD = 1.0

#: How much of the excess saturation is charged to the ceiling. DAMPED at 0.5
#: deliberately: a BuildKit solve is not uniformly CPU-bound -- ``COPY`` steps,
#: the uv cache-mount fetches and the image exports are I/O and network -- so
#: charging the full ratio would over-state the cost of contention.
CONTENTION_SLOPE = 0.5

#: The widest the contention term alone may open the ceiling.
MAX_CONTENTION_MULTIPLIER = 3.0

#: A cold cache re-executes every ``RUN`` layer the warm build served from
#: cache. Doubling is the conservative reading of the ticket's own measurements
#: (a ~404s warm build against builds that had not finished at 1080s).
COLD_CACHE_MULTIPLIER = 2.0

#: An unreadable machine property is charged the same as the bad case it might
#: be hiding. See the module docstring for why outward is the safe direction.
UNKNOWN_CONTENTION_MULTIPLIER = 2.0
UNKNOWN_CACHE_MULTIPLIER = COLD_CACHE_MULTIPLIER

#: At or below this percentage, I/O stalls leave the model ceiling unchanged.
IO_PRESSURE_THRESHOLD_PERCENT = 10.0

#: Charge each unit of excess fraction: 1 + (pct - threshold) / 100 * slope.
IO_PRESSURE_SLOPE = 2.0

#: The widest the I/O term alone may open the ceiling.
MAX_IO_PRESSURE_MULTIPLIER = 2.0

#: An unreadable PSI interface may be hiding stalls, so widen rather than guess.
UNKNOWN_IO_PRESSURE_MULTIPLIER = 1.5

#: Linux exposes I/O stall time here; other platforms may have no PSI interface.
IO_PRESSURE_PATH = Path("/proc/pressure/io")

#: Below this much BuildKit cache the next build is treated as cold. The host's
#: own ``docker builder prune --force --filter until=3h0m0s`` leaves it far
#: under this; a lane that has just built the runtime image leaves it far over.
COLD_CACHE_THRESHOLD_BYTES = 2 * 1024**3

#: How long the cache probe may take before it is abandoned as UNKNOWN. A
#: ceiling derivation must never be the thing that hangs a deploy.
CACHE_PROBE_TIMEOUT_SECONDS = 10

_TOTAL_RE = re.compile(r"^\s*Total:\s*([0-9.]+)\s*([KMGT]?i?B)\s*$", re.IGNORECASE)
_UNIT_BYTES: dict[str, int] = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}

LoadAvgReader = Callable[[], tuple[float, float, float]]
CpuCountReader = Callable[[], int | None]
BuilderCacheReader = Callable[[], EnumBuildCacheState]
IoPressureReader = Callable[[], float | None]


class ModelHostConditions(BaseModel):
    """What was read about the machine, and the multiplier it implies.

    A null machine reading is a READING THAT FAILED, recorded as such, except
    when I/O PSI is explicitly NOT_EXPOSED by the platform. Its state records
    that distinction. Nothing here substitutes a plausible default for a fact
    it could not obtain: a ceiling derived from an invented load average is
    the same class of undetectable wrongness as the constant OMN-18072 replaced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    load1: float | None
    cpu_count: int | None
    saturation: float | None
    cache_state: EnumBuildCacheState
    contention_multiplier: float
    cache_multiplier: float
    io_pressure_some_percent: float | None = None
    io_pressure_state: EnumIoPressureState = EnumIoPressureState.NOT_EXPOSED
    io_pressure_multiplier: float = 1.0

    @property
    def multiplier(self) -> float:
        """The combined widening factor applied to the model-derived ceiling."""
        return (
            self.contention_multiplier
            * self.cache_multiplier
            * self.io_pressure_multiplier
        )

    def describe(self) -> str:
        """One-line, log-ready statement of the host terms and their source."""
        if self.saturation is None:
            load = f"load1 unreadable (x{self.contention_multiplier:.2f})"
        else:
            load = (
                f"load1 {self.load1:.2f} over {self.cpu_count} cpu "
                f"(saturation {self.saturation:.2f}, "
                f"x{self.contention_multiplier:.2f})"
            )
        cache = f"cache {self.cache_state.value} (x{self.cache_multiplier:.2f})"
        if self.io_pressure_some_percent is not None:
            io = f"io psi some {self.io_pressure_some_percent:.2f}%"
        elif self.io_pressure_state is EnumIoPressureState.NOT_EXPOSED:
            io = "io psi not exposed"
        else:
            io = "io psi unreadable"
        io = f"{io} (x{self.io_pressure_multiplier:.2f})"
        return f"{load} + {cache} + {io} => x{self.multiplier:.2f}"


def _default_cpu_count() -> int | None:
    return os.cpu_count()


def _default_io_pressure_reader() -> float | None:
    """Read the larger sustained ``some`` I/O stall percentage from Linux PSI.

    Other failures raise so ``probe_host_conditions`` owns UNKNOWN, just as
    it does for the cache reader.
    """
    try:
        text = IO_PRESSURE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        # No PSI interface is a platform fact, not a reading that failed.
        return None
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] != "some":
            continue
        values = dict(field.split("=", 1) for field in fields[1:])
        averages = (float(values.get("avg60", "")), float(values.get("avg300", "")))
        if any(not 0.0 <= average <= 100.0 for average in averages):
            msg = "I/O PSI some averages are not finite percentages"
            raise ValueError(msg)
        return max(averages)
    msg = "I/O PSI printed no parseable some line"
    raise ValueError(msg)


def _default_builder_cache_reader() -> EnumBuildCacheState:
    """Classify the local BuildKit cache from ``docker builder du``.

    Deliberately reads the TOTAL rather than a per-record breakdown: the
    question this answers is "is there a cache to reuse at all", which is the
    state ``docker builder prune`` changes, and a total is the one line whose
    format is stable across docker versions.

    Raises rather than returning ``UNKNOWN`` so the single catch in
    ``probe_host_conditions`` is the only place a degraded reading is turned
    into a value -- one place to read, one place to change.
    """
    result = subprocess.run(
        ["docker", "builder", "du"],
        capture_output=True,
        text=True,
        check=False,
        timeout=CACHE_PROBE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        msg = f"docker builder du exited {result.returncode}: {result.stderr[:200]}"
        raise RuntimeError(msg)
    for line in result.stdout.splitlines():
        match = _TOTAL_RE.match(line)
        if match is None:
            continue
        scale = _UNIT_BYTES.get(match.group(2).lower())
        if scale is None:
            continue
        total = float(match.group(1)) * scale
        return (
            EnumBuildCacheState.COLD
            if total < COLD_CACHE_THRESHOLD_BYTES
            else EnumBuildCacheState.WARM
        )
    msg = "docker builder du printed no parseable Total: line"
    raise RuntimeError(msg)


def _contention_multiplier(saturation: float | None) -> float:
    if saturation is None:
        return UNKNOWN_CONTENTION_MULTIPLIER
    excess = saturation - CONTENTION_SATURATION_THRESHOLD
    if excess <= 0:
        return 1.0
    return min(MAX_CONTENTION_MULTIPLIER, 1.0 + excess * CONTENTION_SLOPE)


def _cache_multiplier(state: EnumBuildCacheState) -> float:
    if state is EnumBuildCacheState.WARM:
        return 1.0
    if state is EnumBuildCacheState.COLD:
        return COLD_CACHE_MULTIPLIER
    return UNKNOWN_CACHE_MULTIPLIER


def _io_pressure_multiplier(percent: float | None, state: EnumIoPressureState) -> float:
    if state is EnumIoPressureState.UNKNOWN:
        return UNKNOWN_IO_PRESSURE_MULTIPLIER
    if percent is None or percent <= IO_PRESSURE_THRESHOLD_PERCENT:
        return 1.0
    excess = (percent - IO_PRESSURE_THRESHOLD_PERCENT) / 100
    return min(MAX_IO_PRESSURE_MULTIPLIER, 1.0 + excess * IO_PRESSURE_SLOPE)


class _FailedReading:
    """The marker ``_attempt`` returns for a reader that raised."""


_FAILED = _FailedReading()


def _attempt(read: Callable[[], object], what: str) -> object:
    """Call one host reader; a reader that raises becomes ``_FAILED``.

    This is the module's ONE broad catch, so every reader degrades the same
    way and there is one place to read and change it (see the module
    docstring for why a ceiling derivation must never raise).
    """
    try:
        return read()
    except Exception as exc:  # noqa: BLE001 -- see module docstring
        logger.warning("host_conditions: %s unreadable: %s", what, exc)
        return _FAILED


def probe_host_conditions(
    *,
    loadavg_reader: LoadAvgReader | None = None,
    cpu_count_reader: CpuCountReader | None = None,
    builder_cache_reader: BuilderCacheReader | None = None,
    io_pressure_reader: IoPressureReader | None = None,
) -> ModelHostConditions:
    """Read the machine's contention, cache state and I/O pressure. NEVER raises.

    The four readers are keyword seams, not CLI options: a caller that could
    ASSERT its own host conditions could assert an idle machine and buy itself
    a wider ceiling, which is the shape of caller-supplied premise the
    prod-promotion gate had deleted from it under OMN-18319. Nothing reaches
    these parameters from argv; only a test supplies them.
    """
    load1: float | None = None
    cpu_count: int | None = None
    saturation: float | None = None

    read_load = loadavg_reader or os.getloadavg
    read_cpus = cpu_count_reader or _default_cpu_count
    read_cache = builder_cache_reader or _default_builder_cache_reader
    read_io = io_pressure_reader or _default_io_pressure_reader

    load_read = _attempt(lambda: float(read_load()[0]), "load average")
    load1 = load_read if isinstance(load_read, float) else None

    def _read_cpu_count() -> int | None:
        raw_cpus = read_cpus()
        return int(raw_cpus) if raw_cpus else None

    cpus_read = _attempt(_read_cpu_count, "cpu count")
    cpu_count = cpus_read if isinstance(cpus_read, int) else None

    if load1 is not None and cpu_count:
        saturation = load1 / cpu_count

    cache_read = _attempt(read_cache, "BuildKit cache state")
    cache_state = (
        cache_read
        if isinstance(cache_read, EnumBuildCacheState)
        else EnumBuildCacheState.UNKNOWN
    )

    io_read = _attempt(read_io, "I/O pressure")
    io_pressure_some_percent: float | None = None
    if io_read is _FAILED:
        io_pressure_state = EnumIoPressureState.UNKNOWN
    elif io_read is None:
        io_pressure_state = EnumIoPressureState.NOT_EXPOSED
    elif isinstance(io_read, (int, float)) and not isinstance(io_read, bool):
        io_pressure_some_percent = float(io_read)
        io_pressure_state = EnumIoPressureState.READ
    else:
        io_pressure_state = EnumIoPressureState.UNKNOWN

    return ModelHostConditions(
        load1=load1,
        cpu_count=cpu_count,
        saturation=saturation,
        cache_state=cache_state,
        contention_multiplier=_contention_multiplier(saturation),
        cache_multiplier=_cache_multiplier(cache_state),
        io_pressure_some_percent=io_pressure_some_percent,
        io_pressure_state=io_pressure_state,
        io_pressure_multiplier=_io_pressure_multiplier(
            io_pressure_some_percent, io_pressure_state
        ),
    )
