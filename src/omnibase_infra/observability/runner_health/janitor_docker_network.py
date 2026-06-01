# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bounded Docker network janitor — declared-ownership reclaim only.

The janitor never runs a blanket ``docker network prune``. It inspects each
network, classifies it against a *declared ownership contract*, and removes
ONLY networks that simultaneously:

  1. match a declared ownership rule's name pattern,
  2. are older than that rule's ``min_age_seconds``, and
  3. have zero attached containers.

Every other network is preserved. Unknown ownership defaults to preserve — a
naming mistake can never become destructive, and an active lane (any network
with attached containers) is never touched.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from omnibase_infra.observability.runner_health.enum_network_disposition import (
    EnumNetworkDisposition,
)
from omnibase_infra.observability.runner_health.model_network_decision import (
    ModelNetworkDecision,
)
from omnibase_infra.observability.runner_health.model_network_info import (
    ModelNetworkInfo,
)
from omnibase_infra.observability.runner_health.model_network_janitor_result import (
    ModelNetworkJanitorResult,
)
from omnibase_infra.observability.runner_health.model_network_ownership_rule import (
    DEFAULT_OWNERSHIP_RULES,
    ModelNetworkOwnershipRule,
)


def classify_network(
    network: ModelNetworkInfo,
    rules: Sequence[ModelNetworkOwnershipRule],
    now: datetime,
) -> ModelNetworkDecision:
    """Decide the disposition of a single network under the ownership contract.

    Pure function — no I/O. Order of precedence is safety-first:

    builtin > unknown-ownership > active (attached containers) > age-unknown
    > too-young > reclaim.
    """
    if network.is_builtin:
        return ModelNetworkDecision(
            network_ref=network.network_ref,
            name=network.name,
            disposition=EnumNetworkDisposition.PRESERVE_BUILTIN,
            container_count=network.container_count,
            reason="Docker builtin network is never a janitor target",
        )

    matched: ModelNetworkOwnershipRule | None = None
    for rule in rules:
        if rule.matches_name(network.name):
            matched = rule
            break

    if matched is None:
        return ModelNetworkDecision(
            network_ref=network.network_ref,
            name=network.name,
            disposition=EnumNetworkDisposition.PRESERVE_UNKNOWN_OWNERSHIP,
            container_count=network.container_count,
            reason="No declared ownership rule matched — preserve, never delete",
        )

    # Owned. Active lanes (attached containers) are never reclaimed, even if
    # they match a rule and are old — attachment means the network is in use.
    if network.container_count > 0:
        return ModelNetworkDecision(
            network_ref=network.network_ref,
            name=network.name,
            disposition=EnumNetworkDisposition.PRESERVE_ACTIVE,
            matched_rule=matched.name,
            container_count=network.container_count,
            reason=(
                f"Owned by '{matched.name}' but has "
                f"{network.container_count} attached container(s) — active lane"
            ),
        )

    if network.created_at is None:
        return ModelNetworkDecision(
            network_ref=network.network_ref,
            name=network.name,
            disposition=EnumNetworkDisposition.PRESERVE_AGE_UNKNOWN,
            matched_rule=matched.name,
            container_count=network.container_count,
            reason="Owned but creation time unknown — preserve, never guess age",
        )

    age_seconds = int((now - network.created_at).total_seconds())
    if age_seconds < matched.min_age_seconds:
        return ModelNetworkDecision(
            network_ref=network.network_ref,
            name=network.name,
            disposition=EnumNetworkDisposition.PRESERVE_TOO_YOUNG,
            matched_rule=matched.name,
            container_count=network.container_count,
            age_seconds=age_seconds,
            reason=(
                f"Owned by '{matched.name}' but age {age_seconds}s "
                f"< threshold {matched.min_age_seconds}s"
            ),
        )

    return ModelNetworkDecision(
        network_ref=network.network_ref,
        name=network.name,
        disposition=EnumNetworkDisposition.RECLAIM,
        matched_rule=matched.name,
        container_count=network.container_count,
        age_seconds=age_seconds,
        reason=(
            f"Owned by '{matched.name}', idle, age {age_seconds}s "
            f">= threshold {matched.min_age_seconds}s — reclaimable"
        ),
    )


class JanitorDockerNetwork:
    """Bounded Docker network reclaimer for a single runner host.

    Inspection and removal go over SSH (the runner host is ``.201``). The
    janitor is safe by construction: the removal command operates only on the
    explicit set of network IDs that ``classify_network`` marked ``RECLAIM``.
    It never issues ``docker network prune``.
    """

    def __init__(
        self,
        runner_host: str,
        rules: Sequence[ModelNetworkOwnershipRule] = DEFAULT_OWNERSHIP_RULES,
    ) -> None:
        self._runner_host = runner_host
        self._rules = tuple(rules)

    async def _fetch_networks(self) -> list[ModelNetworkInfo]:
        """List networks on the host via SSH + ``docker network inspect``.

        Emits one tab-separated line per network:
        ``<id>\\t<name>\\t<created>\\t<container_count>``.
        """
        cmd = (
            "for nid in $(docker network ls -q); do "
            "docker network inspect "
            '--format "{{.Id}}\t{{.Name}}\t{{.Created}}\t{{len .Containers}}" '
            '"$nid" 2>/dev/null; '
            "done"
        )
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            self._runner_host,
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return []
        return [
            info
            for line in stdout.decode().splitlines()
            if (info := _parse_network_line(line)) is not None
        ]

    async def _remove_networks(self, network_ids: Sequence[str]) -> list[str]:
        """Remove exactly the given network IDs via SSH. Returns error lines.

        Each ``docker network rm`` is independent: a failure on one (e.g. a
        network that re-acquired a container between inspection and removal)
        does not abort the rest, and Docker itself refuses to delete a network
        with attached endpoints — a second safety net under the classifier.
        """
        if not network_ids:
            return []
        quoted = " ".join(_shell_single_quote(nid) for nid in network_ids)
        cmd = (
            f"for nid in {quoted}; do "
            'docker network rm "$nid" >/dev/null 2>&1 '
            '|| echo "rm_failed:$nid"; '
            "done"
        )
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            self._runner_host,
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return [
            line.strip()
            for line in stdout.decode().splitlines()
            if line.startswith("rm_failed:")
        ]

    async def run(
        self,
        correlation_id: UUID,
        dry_run: bool = True,
        now: datetime | None = None,
    ) -> ModelNetworkJanitorResult:
        """Inspect, classify, and (unless ``dry_run``) reclaim networks."""
        moment = now or datetime.now(tz=UTC)
        networks = await self._fetch_networks()
        decisions = tuple(classify_network(n, self._rules, moment) for n in networks)
        reclaim_ids = tuple(
            d.network_ref
            for d in decisions
            if d.disposition is EnumNetworkDisposition.RECLAIM
        )

        reclaimed: tuple[str, ...] = ()
        errors: tuple[str, ...] = ()
        if not dry_run and reclaim_ids:
            error_lines = await self._remove_networks(reclaim_ids)
            failed = {line.removeprefix("rm_failed:") for line in error_lines}
            reclaimed = tuple(nid for nid in reclaim_ids if nid not in failed)
            errors = tuple(error_lines)

        return ModelNetworkJanitorResult(
            correlation_id=correlation_id,
            ran_at=moment,
            host=self._runner_host,
            dry_run=dry_run,
            decisions=decisions,
            reclaimed=reclaimed,
            reclaim_errors=errors,
        )


_BUILTIN_NETWORK_NAMES = frozenset({"bridge", "host", "none"})


def _shell_single_quote(value: str) -> str:
    """Single-quote a value for safe embedding in a shell command."""
    return "'" + value.replace("'", "'\\''") + "'"


def _parse_network_line(line: str) -> ModelNetworkInfo | None:
    """Parse one ``inspect`` line into ModelNetworkInfo, or None if malformed."""
    parts = line.split("\t")
    if len(parts) != 4:
        return None
    network_ref, name, created_raw, count_raw = (p.strip() for p in parts)
    if not network_ref or not name:
        return None
    try:
        container_count = int(count_raw)
    except ValueError:
        container_count = 0
    return ModelNetworkInfo(
        network_ref=network_ref,
        name=name,
        created_at=_parse_docker_timestamp(created_raw),
        container_count=container_count,
        is_builtin=name in _BUILTIN_NETWORK_NAMES,
    )


def _parse_docker_timestamp(raw: str) -> datetime | None:
    """Parse a Docker ``.Created`` RFC3339 timestamp into an aware UTC datetime.

    Docker emits nanosecond precision (e.g.
    ``2026-06-01T17:29:45.123456789Z``) which ``datetime.fromisoformat`` can
    reject; truncate fractional seconds to microseconds and normalise ``Z``.
    """
    if not raw:
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # Truncate over-long fractional seconds (Docker nanoseconds -> micros).
    if "." in text:
        head, _, tail = text.partition(".")
        digits = ""
        rest = ""
        for i, ch in enumerate(tail):
            if ch.isdigit():
                digits += ch
            else:
                rest = tail[i:]
                break
        text = f"{head}.{digits[:6]}{rest}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = ["JanitorDockerNetwork", "classify_network"]
