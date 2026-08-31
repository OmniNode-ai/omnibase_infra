# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17150 defect 1 — the cold-initdb race the migration one-shots lose.

THE DEFECT
----------
``intelligence-migration`` and ``forward-migration`` are both one-shots, both
``restart: "no"``, and both start behind the identical
``depends_on: postgres: {condition: service_healthy}``. On a cold volume only
one of them survives. Measured on ``omnibase-infra-lakshman`` 2026-08-31,
deterministic across three clean boots (Docker Compose 5.1.0):

    postgres started                       16:31:52.157
    forward-migration started              16:31:57.876  -> ran 16s, exit 0
    intelligence-migration started         16:31:57.879  -> exit 2 after 150ms
    postgres first reported healthy        16:32:27

    psql: error: connection to server at "postgres", port 5432 failed:
    Connection refused

Two independent causes, and this module pins the fix for both.

CAUSE 1 — the healthcheck answers for the wrong server. The postgres image runs
a TEMPORARY server during its initdb / init-script phase, started with
``listen_addresses=''``: unix socket only, no TCP listener. ``pg_isready`` with
no ``-h`` uses the socket, so it reports the container HEALTHY against that
temporary server. Compose releases every ``service_healthy`` dependent, the
temporary server is then stopped and the real one started, and every connection
made in that window is refused. The window is wide on this stack because the
whole ``docker/migrations/forward`` tree is bind-mounted into
``/docker-entrypoint-initdb.d`` and therefore runs inside it.

CAUSE 2 — only one of the two one-shots retried. ``run-forward-migrations.sh``
has carried a wait loop since OMN-13062; ``run-intelligence-migrations.sh`` had
none, and its first statement swallows errors (``2>/dev/null || true``), so a
refusal produced an empty result, the script proceeded to ``CREATE DATABASE``,
and exited 2. With ``restart: "no"`` there was no second attempt, and
``omninode-runtime``'s ``service_completed_successfully`` dependency then held
the entire runtime tier behind a container that had already given up.

The documented workaround was "run ``docker compose up -d`` a second time". That
is the kind of thing that becomes folklore, so it is fixed rather than written
down: the healthcheck now distinguishes the two servers, and the one-shot now
retries like its sibling.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

FORWARD_RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
INTELLIGENCE_RUNNER = REPO_ROOT / "scripts" / "run-intelligence-migrations.sh"
POSTGRES_CATALOG = REPO_ROOT / "docker" / "catalog" / "services" / "postgres.yaml"

# Every compose file that declares a postgres service with a healthcheck other
# services gate on. A new lane file must be added here — that is the point of
# listing them rather than globbing: a lane added without this fix should fail
# this test rather than silently inherit the race.
COMPOSE_FILES = (
    REPO_ROOT / "docker" / "docker-compose.infra.yml",
    REPO_ROOT / "docker" / "docker-compose.judge.yml",
    REPO_ROOT / "docker" / "docker-compose.lakshman.yml",
)

# A cold initdb on this stack also runs the full forward-migration tree from
# /docker-entrypoint-initdb.d. 10s (the previous value) covered none of it.
# The floor is deliberately well under the value actually shipped, so ordinary
# tuning does not require touching this test — only a regression toward the
# old, race-producing value does.
COLD_INITDB_START_PERIOD_FLOOR_S = 120


def _construct_compose_value(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    """Passthrough constructor for Docker Compose ``!override`` / ``!reset``."""
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that unwraps compose merge/override tags (OMN-13772)."""


_ComposeLoader.add_constructor("!override", _construct_compose_value)
_ComposeLoader.add_constructor("!reset", _construct_compose_value)


def _postgres_healthcheck(compose_file: Path) -> dict[str, Any]:
    raw = compose_file.read_text(encoding="utf-8")
    # _ComposeLoader extends SafeLoader; the extra constructors only unwrap
    # compose merge/override tags.
    compose = yaml.load(raw, Loader=_ComposeLoader)  # noqa: S506
    assert isinstance(compose, dict), compose_file
    postgres = compose["services"]["postgres"]
    healthcheck = postgres.get("healthcheck")
    assert isinstance(healthcheck, dict), (
        f"{compose_file.name}: postgres declares no healthcheck, so every "
        "`condition: service_healthy` dependent on it is unguarded"
    )
    return healthcheck


def _duration_seconds(value: object) -> int:
    """Parse the compose duration forms this repo actually uses (``180s``)."""
    text = str(value).strip()
    match = re.fullmatch(r"(\d+)(s|m)?", text)
    assert match is not None, f"unparseable compose duration: {value!r}"
    seconds = int(match.group(1))
    return seconds * 60 if match.group(2) == "m" else seconds


@pytest.mark.unit
@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda p: p.name)
def test_postgres_healthcheck_probes_tcp_not_the_unix_socket(
    compose_file: Path,
) -> None:
    """The discriminator between the temporary init server and the real one.

    ``pg_isready`` with no ``-h`` connects over the local unix socket, which the
    temporary initdb server DOES serve. Only a TCP probe can distinguish them,
    because that server runs with ``listen_addresses=''``.

    Mutation check: drop the ``-h`` and this fails, naming the cause.
    """
    healthcheck = _postgres_healthcheck(compose_file)
    test = healthcheck["test"]
    command = " ".join(test) if isinstance(test, list) else str(test)

    assert "pg_isready" in command, (
        f"{compose_file.name}: postgres healthcheck no longer runs pg_isready: "
        f"{command!r}"
    )
    assert re.search(r"pg_isready[^|;]*\s-h\s", command), (
        f"{compose_file.name}: the postgres healthcheck probes the local unix "
        "socket, which the temporary initdb server also serves. It will report "
        "HEALTHY before the real server accepts TCP, releasing every "
        "`condition: service_healthy` dependent into a refused connection "
        f"(OMN-17150 defect 1). Pass -h to force TCP.\n  found: {command!r}"
    )
    # The -h value must be the compose SERVICE NAME, not a literal address.
    # A loopback literal is functionally identical and was the first thing
    # written here; `TestDockerNetworkSecurity::test_self_contained_infrastructure`
    # correctly rejected it (hardcoded addresses in compose are a portability
    # defect). Asserted here so the two gates cannot be satisfied one at a time.
    assert not re.search(r"pg_isready[^|;]*-h\s+\d+\.\d+\.\d+\.\d+", command), (
        f"{compose_file.name}: the postgres healthcheck probes a hardcoded IP. "
        "Use the compose service name — it forces the same TCP path and stays "
        f"portable across lanes.\n  found: {command!r}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("compose_file", COMPOSE_FILES, ids=lambda p: p.name)
def test_postgres_healthcheck_start_period_covers_a_cold_initdb(
    compose_file: Path,
) -> None:
    """A still-initialising server must stay ``starting``, not flip UNHEALTHY.

    With a TCP probe and the old 10s start_period, a cold volume would be marked
    UNHEALTHY well before initdb finished — trading a false-healthy for a
    false-unhealthy, which fails the dependents just as hard.
    """
    healthcheck = _postgres_healthcheck(compose_file)
    start_period = healthcheck.get("start_period")
    assert start_period is not None, (
        f"{compose_file.name}: postgres healthcheck sets no start_period, so it "
        "inherits the x-healthcheck-defaults value sized for a warm container"
    )
    seconds = _duration_seconds(start_period)
    assert seconds >= COLD_INITDB_START_PERIOD_FLOOR_S, (
        f"{compose_file.name}: postgres start_period is {seconds}s, below the "
        f"{COLD_INITDB_START_PERIOD_FLOOR_S}s floor for a cold initdb. On this "
        "stack initdb also runs the entire docker/migrations/forward tree from "
        "/docker-entrypoint-initdb.d, so a short start_period marks a healthy-"
        "but-slow server UNHEALTHY and fails every dependent (OMN-17150)."
    )


@pytest.mark.unit
def test_postgres_catalog_manifest_matches_the_compose_healthcheck() -> None:
    """The catalog is the declared shape; a fix applied to only one drifts."""
    catalog = yaml.safe_load(POSTGRES_CATALOG.read_text(encoding="utf-8"))
    command = str(catalog["healthcheck"]["test"])
    assert re.search(r"pg_isready[^|;]*\s-h\s", command), (
        "docker/catalog/services/postgres.yaml still declares a socket-based "
        f"pg_isready while the compose lanes probe TCP: {command!r}"
    )
    assert (
        int(catalog["healthcheck"]["start_period_s"])
        >= COLD_INITDB_START_PERIOD_FLOOR_S
    ), "the catalog's postgres start_period is below the cold-initdb floor"


@pytest.mark.unit
def test_intelligence_migration_waits_for_postgres_before_its_first_statement() -> None:
    """The one-shot must retry, not die on the first refusal.

    Ordering matters as much as presence: the wait has to precede the
    ``CREATE DATABASE``, whose preceding probe swallows errors and therefore
    cannot itself detect that Postgres was never reachable.
    """
    text = INTELLIGENCE_RUNNER.read_text(encoding="utf-8")

    assert "PG_WAIT_RETRIES" in text, (
        "run-intelligence-migrations.sh has no bounded Postgres wait. It starts "
        "at the same instant as run-forward-migrations.sh behind the same "
        "service_healthy dependency, and dies on the first refused connection "
        "during a cold initdb (OMN-17150 defect 1)."
    )

    # Comment lines are stripped first: this section's own rationale comment
    # names CREATE DATABASE, and matching that would compare the wait loop
    # against a sentence about it rather than against the statement.
    executable = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    wait_at = executable.find("Waiting for Postgres to accept connections")
    create_at = executable.find("CREATE DATABASE")
    assert wait_at != -1, "the wait loop's log line is missing"
    assert create_at != -1, "CREATE DATABASE is missing — did the script change?"
    assert wait_at < create_at, (
        "the Postgres wait loop runs AFTER the first CREATE DATABASE, so the "
        "statement that actually failed on the lakshman lane is still unguarded"
    )

    assert re.search(r'-ge\s+"?\$\{?PG_WAIT_RETRIES', text), (
        "the wait loop is unbounded — it must abort loudly after "
        "PG_WAIT_RETRIES attempts rather than hanging the lane's bring-up"
    )
    assert "sleep" in text, "the wait loop must back off between attempts"


@pytest.mark.unit
def test_both_migration_one_shots_share_one_wait_contract() -> None:
    """Parity, mechanically. The defect WAS the asymmetry.

    Two one-shots, one compose dependency, one race — and a retry in only one of
    them. Pinning the shared contract is what stops the two drifting apart
    again; a future change to the forward runner's budget that skips the
    intelligence runner fails here.
    """
    forward = FORWARD_RUNNER.read_text(encoding="utf-8")
    intelligence = INTELLIGENCE_RUNNER.read_text(encoding="utf-8")

    pattern = r'PG_WAIT_RETRIES="\$\{PG_WAIT_RETRIES:-(\d+)\}"'
    forward_default = re.search(pattern, forward)
    intelligence_default = re.search(pattern, intelligence)
    assert forward_default is not None, (
        "run-forward-migrations.sh no longer declares a PG_WAIT_RETRIES default"
    )
    assert intelligence_default is not None, (
        "run-intelligence-migrations.sh no longer declares a PG_WAIT_RETRIES default"
    )
    assert forward_default.group(1) == intelligence_default.group(1), (
        "the two migration one-shots disagree on how long to wait for Postgres "
        f"({FORWARD_RUNNER.name}={forward_default.group(1)}, "
        f"{INTELLIGENCE_RUNNER.name}={intelligence_default.group(1)}). They "
        "race the same initdb behind the same dependency; a difference here is "
        "the shape of the original defect, where one had no budget at all."
    )

    for name, text in (
        (FORWARD_RUNNER.name, forward),
        (INTELLIGENCE_RUNNER.name, intelligence),
    ):
        assert "Postgres not ready after" in text, (
            f"{name}: exhausting the wait budget must say so on stderr and "
            "exit non-zero — a silent give-up is what left the lakshman lane "
            "looking like a wiring bug"
        )
