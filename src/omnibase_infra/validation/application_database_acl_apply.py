# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Sanctioned apply path for the generated application-database ACL matrix.

OMN-15355. The matrix generator and its PostgreSQL 16 proof already produce and
exercise the GRANT/REVOKE program; what did not exist was a governed way to put
that program onto a *live* database. Hand-run ``psql`` is never that way.

This module is the gate, not the SQL. Every refusal below is fail-closed, and
each one exists because the corresponding mistake is silent:

* a matrix that is ``BLOCKED`` still renders *something*, and applying it would
  encode the blockers as live privilege;
* a ``synthetic_proof`` matrix is built from fixtures and describes no real
  deployment;
* an apply with no durable pre-change snapshot has no rollback, only a
  handwritten approximation of one, which the ticket forbids by name;
* a connection probe that could not be run reads exactly like a probe that
  passed, so an unverifiable principal fails the run rather than passing it.

Credentials never appear here. A probe carries the NAME of the environment
variable its DSN is read from and nothing else, so a report, a log line, or a
traceback cannot leak a secret value.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path

from omnibase_infra.validation.application_database_acl import (
    render_application_database_acl_sql,
)
from omnibase_infra.validation.enums.enum_acl_connection_probe_kind import (
    EnumAclConnectionProbeKind,
)
from omnibase_infra.validation.enums.enum_application_database_acl_authorization_scope import (
    EnumApplicationDatabaseAclAuthorizationScope,
)
from omnibase_infra.validation.enums.enum_application_database_acl_render_phase import (
    EnumApplicationDatabaseAclRenderPhase,
)
from omnibase_infra.validation.models.model_acl_apply_consent import (
    ModelAclApplyConsent,
)
from omnibase_infra.validation.models.model_acl_apply_report import ModelAclApplyReport
from omnibase_infra.validation.models.model_acl_connection_probe import (
    ModelAclConnectionProbe,
    resolve_role_dsn_env_name,
)
from omnibase_infra.validation.models.model_application_database_acl_matrix import (
    ModelApplicationDatabaseAclMatrix,
)

#: The two people who may approve a live privilege change (omni_home rule 22).
APPROVERS: frozenset[str] = frozenset({"operator", "jake"})

#: One captured pre-change ACL state: a section name to its recorded rows.
AclSnapshot = Mapping[str, object]

_CONSENT_MARKER = "OPERATOR-CONSENT"
_APPROVED_SCOPE_MARKER = "APPROVED SCOPE:"
_OUT_OF_SCOPE_MARKER = "OUT OF SCOPE:"
_LANE_RE = re.compile(r"\blane=([A-Za-z0-9_.-]+)")
_APPROVED_BY_RE = re.compile(r"\bapproved_by=([A-Za-z0-9_.-]+)")
_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)")


class AclApplyRefusalError(RuntimeError):
    """A fail-closed refusal from the ACL apply path."""


def resolve_consent_citation(
    citation: str,
    *,
    ticket: str,
    ledger_root: Path,
) -> ModelAclApplyConsent:
    """Resolve ``<ledger>@<row timestamp>`` or ``<ledger>:<line>`` to a consent row.

    Refuses fail-closed on every ambiguity: a malformed citation, a line past
    the end of the file, a timestamp no row opens with, a timestamp more than
    one row opens with, a row that is not a consent row, an approver outside the
    named pair, a row missing either scope half, or an APPROVED SCOPE that does
    not name the ticket being applied.

    THE TIMESTAMP FORM IS THE DURABLE ONE (OMN-18620). A line number is only
    true until the ledger is next rolled: a cap-crossing roll on 2026-09-17
    removed 926 lines from the top of the live file and moved a consent row from
    ``:4367`` to ``:3441``. A row timestamp travels with the row, into the
    archive when it is rolled, so it survives any number of rolls. The line form
    is kept because citations already written must not be invalidated by the
    change that introduces the replacement.
    """
    # `@` FIRST, and on the LAST `@`, because the line form's split is
    # `rpartition(":")` and an ISO-8601 timestamp is full of colons -- reading a
    # timestamp citation as a line citation would split it mid-timestamp and
    # report a malformed-citation refusal for a perfectly good citation.
    ledger_path, at_sign, raw_stamp = citation.rpartition("@")
    if at_sign and ledger_path:
        return _resolve_by_stamp(
            ledger_path, raw_stamp, ticket=ticket, ledger_root=ledger_root
        )
    ledger_path, separator, raw_line = citation.rpartition(":")
    if not separator or not raw_line.isdigit() or not ledger_path:
        raise AclApplyRefusalError(
            "consent citation must be '<ledger>@<row timestamp>' or "
            f"'<ledger>:<line>', got {citation!r}"
        )
    line_number = int(raw_line)
    path = Path(ledger_path)
    if not path.is_absolute():
        path = ledger_root / path
    if not path.is_file():
        raise AclApplyRefusalError(f"consent citation names no readable ledger: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if line_number < 1 or line_number > len(lines):
        raise AclApplyRefusalError(
            f"consent citation names line {line_number}, "
            f"which is past the end of {path} ({len(lines)} lines)"
        )
    row = lines[line_number - 1]
    return _validate_consent_row(
        row, f"{path}:{line_number}", path, line_number, ticket=ticket
    )


def _is_real_file_inside(candidate: Path, directory: Path) -> bool:
    """Whether ``candidate`` is a regular file that really lives in ``directory``.

    THE GLOB RESTRICTS THE NAME, NOT THE INODE. A symlink called
    ``<ledger stem>_<date>-split.md`` matches the pattern, and ``read_text``
    follows symlinks, so without this a link planted in the archive directory
    could point at any file on the host and answer a consent citation with
    whatever that file contains -- authorising a live GRANT or REVOKE. The name
    is attacker-choosable; the target must not be.

    Four conditions, each load-bearing: the DIRECTORY is not itself a symlink
    (otherwise every entry in it resolves consistently and the check passes while
    reading a tree somewhere else); the entry is not a symlink; it is a regular
    file, not a fifo, device or directory; and it resolves to something whose
    parent really is this directory.

    What this does NOT claim: the archive is not a trusted store. Anyone who can
    plant a file in it can also append a row to the live ledger, which is read
    with no such check because it is the file the citation names. This closes one
    specific vector and nothing wider.
    """
    try:
        if directory.is_symlink():
            return False
        if candidate.is_symlink() or not candidate.is_file():
            return False
        return candidate.resolve().parent == directory.resolve()
    except OSError:
        return False


def _rows_opening_with(
    lines: list[str], stamp: str, source: Path
) -> list[tuple[Path, int, str]]:
    """Every ``(file, 1-based line, row)`` whose LEADING timestamp is ``stamp``.

    Leading, not "contains": rows routinely quote other rows' timestamps --
    every citation in this fleet does -- and matching those would resolve a
    citation to a row that merely mentions the one meant.

    The file and line come back with the row so the resolved consent record can
    still name WHERE it was found, which is the archive file when the row has
    been rolled. A record that could not say that would report an archived row
    as if it were still in the live ledger.
    """
    out: list[tuple[Path, int, str]] = []
    for index, line in enumerate(lines, start=1):
        match = _TIMESTAMP_RE.match(line.strip())
        if match is not None and match.group(1) == stamp:
            out.append((source, index, line))
    return out


def _resolve_by_stamp(
    ledger_path: str,
    stamp: str,
    *,
    ticket: str,
    ledger_root: Path,
) -> ModelAclApplyConsent:
    """Resolve a ``@<row timestamp>`` citation, searching the archive too."""
    if _TIMESTAMP_RE.match(stamp) is None:
        raise AclApplyRefusalError(
            f"consent citation timestamp must be ISO-8601 UTC to the second, "
            f"got {stamp!r}"
        )
    path = Path(ledger_path)
    if not path.is_absolute():
        path = ledger_root / path
    if not path.is_file():
        raise AclApplyRefusalError(f"consent citation names no readable ledger: {path}")
    located = _rows_opening_with(
        path.read_text(encoding="utf-8").splitlines(), stamp, path
    )
    searched = [str(path)]
    archive_dir = path.parent / "archive"
    if not located and archive_dir.is_dir():
        # A rolled row is still a real consent row, and this is the whole point
        # of the timestamp form: the row moved into the archive and its
        # timestamp went with it.
        # SCOPED TO THE ROLL'S OWN FILENAME SHAPE, not `*.md`. A bare glob
        # would read every markdown file in the directory, so anything that
        # could land a file there -- a stray doc, a partial write, a crafted
        # name -- could carry a row with the target timestamp and the required
        # fields and authorise a live GRANT/REVOKE nobody consented to. The
        # names accepted are the ones `ledger_lock.py` itself writes beside this
        # ledger: `<ledger stem>_<date>-split.md`.
        for archive in sorted(archive_dir.glob(f"{path.stem}_*-split.md")):
            if not _is_real_file_inside(archive, archive_dir):
                continue
            searched.append(str(archive))
            located.extend(
                _rows_opening_with(
                    archive.read_text(encoding="utf-8").splitlines(),
                    stamp,
                    archive,
                )
            )
    if not located:
        raise AclApplyRefusalError(
            f"no row in {', '.join(searched)} opens with the timestamp {stamp}"
        )
    if len(located) > 1:
        # AMBIGUITY IS A REFUSAL, never a pick. Two lanes can append inside the
        # same second, so a row timestamp is not guaranteed unique, and choosing
        # one of several would mean applying a live privilege change against a
        # row nobody cited.
        raise AclApplyRefusalError(
            f"{len(located)} rows open with the timestamp {stamp}, so the "
            f"citation does not name one row; cite '<ledger>:<line>' instead"
        )
    found_path, found_line, row = located[0]
    return _validate_consent_row(
        row, f"{path}@{stamp}", found_path, found_line, ticket=ticket
    )


def _validate_consent_row(
    row: str, ref: str, path: Path, line_number: int, *, ticket: str
) -> ModelAclApplyConsent:
    """The checks a resolved row must pass, whichever form cited it.

    Shared so the two citation forms cannot drift into different standards --
    a form that resolved rows while skipping these would be an authorisation
    bypass rather than a convenience.
    """
    if _CONSENT_MARKER not in row:
        raise AclApplyRefusalError(f"{ref} is not an OPERATOR-CONSENT row")
    approved_by_match = _APPROVED_BY_RE.search(row)
    if approved_by_match is None or approved_by_match.group(1) not in APPROVERS:
        named = approved_by_match.group(1) if approved_by_match else "<absent>"
        raise AclApplyRefusalError(
            f"{ref} names approver {named!r}; a live privilege "
            f"change requires one of {sorted(APPROVERS)}"
        )
    if _APPROVED_SCOPE_MARKER not in row:
        raise AclApplyRefusalError(f"{ref} carries no APPROVED SCOPE")
    if _OUT_OF_SCOPE_MARKER not in row:
        raise AclApplyRefusalError(
            f"{ref} carries no OUT OF SCOPE half; the "
            "out-of-scope list is what bounds the grant"
        )
    approved_scope = row.split(_APPROVED_SCOPE_MARKER, 1)[1].split(
        _OUT_OF_SCOPE_MARKER, 1
    )[0]
    out_of_scope = row.split(_OUT_OF_SCOPE_MARKER, 1)[1]
    if ticket not in approved_scope:
        raise AclApplyRefusalError(f"{ref} APPROVED SCOPE does not name {ticket}")
    lane_match = _LANE_RE.search(row)
    timestamp_match = _TIMESTAMP_RE.match(row.strip())
    if timestamp_match is None:
        raise AclApplyRefusalError(f"{ref} does not begin with a UTC timestamp")
    return ModelAclApplyConsent(
        ledger_path=str(path),
        line_number=line_number,
        lane=lane_match.group(1) if lane_match else "<absent>",
        approved_by=approved_by_match.group(1),
        recorded_at=datetime.fromisoformat(
            timestamp_match.group(1).replace("Z", "+00:00")
        ),
        approved_scope=approved_scope.strip(),
        out_of_scope=out_of_scope.strip(),
    )


def assert_matrix_appliable(
    matrix: ModelApplicationDatabaseAclMatrix,
    *,
    phase: EnumApplicationDatabaseAclRenderPhase = (
        EnumApplicationDatabaseAclRenderPhase.FULL
    ),
) -> None:
    """Refuse any matrix that must not reach a live database."""
    if (
        matrix.authorization_scope
        is not EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT
    ):
        raise AclApplyRefusalError(
            "refusing to apply a "
            f"{matrix.authorization_scope.value} matrix to a live database; "
            "only a deployment-scoped matrix describes a real deployment"
        )
    if phase is EnumApplicationDatabaseAclRenderPhase.SCAFFOLD:
        status, blockers = matrix.scaffold_status, matrix.scaffold_blockers
    else:
        status, blockers = matrix.status, matrix.blockers
    if status != "READY":
        listed = "\n  ".join(blockers) or "<none recorded>"
        raise AclApplyRefusalError(
            f"refusing to apply a {status} {phase.value} matrix; blockers:\n  {listed}"
        )


def plan_connection_probes(
    matrix: ModelApplicationDatabaseAclMatrix,
) -> tuple[ModelAclConnectionProbe, ...]:
    """Plan one positive probe per allowed pair and a negative for every other.

    The PUBLIC negative is unconditional per database: revoking CONNECT from
    PUBLIC is the whole point of the change window, and a run that does not
    assert it has not proven it.
    """
    universe: set[str] = set()
    for principals in matrix.declared_principals.values():
        universe.update(principals)
    for principals in matrix.allowed_connect_principals.values():
        universe.update(principals)
    probes: list[ModelAclConnectionProbe] = []
    for database in matrix.required_connect_databases:
        allowed = tuple(matrix.allowed_connect_principals.get(database, ()))
        for principal in sorted(allowed):
            probes.append(
                ModelAclConnectionProbe(
                    database=database,
                    principal=principal,
                    kind=EnumAclConnectionProbeKind.POSITIVE,
                )
            )
        for principal in sorted(universe - set(allowed)):
            probes.append(
                ModelAclConnectionProbe(
                    database=database,
                    principal=principal,
                    kind=EnumAclConnectionProbeKind.NEGATIVE,
                )
            )
        probes.append(
            ModelAclConnectionProbe(
                database=database,
                kind=EnumAclConnectionProbeKind.NEGATIVE_PUBLIC,
            )
        )
    return tuple(probes)


def _write_snapshot(snapshot_path: Path, snapshot: AclSnapshot) -> None:
    try:
        snapshot_path.write_text(
            json.dumps(dict(snapshot), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise AclApplyRefusalError(
            "refusing to mutate: the pre-change ACL snapshot could not be "
            f"written to {snapshot_path} ({error}); without a durable snapshot "
            "there is no rollback artifact"
        ) from error


def apply_application_database_acl(
    matrix: ModelApplicationDatabaseAclMatrix,
    *,
    consent_citation: str,
    ticket: str,
    ledger_root: Path,
    snapshot_path: Path,
    capture_snapshot: Callable[[ModelApplicationDatabaseAclMatrix], AclSnapshot],
    execute_sql: Callable[[str], None],
    run_probe: Callable[[ModelAclConnectionProbe], bool],
    restore_snapshot: Callable[[Path], None] | None = None,
    phase: EnumApplicationDatabaseAclRenderPhase = (
        EnumApplicationDatabaseAclRenderPhase.FULL
    ),
    execute: bool = False,
) -> ModelAclApplyReport:
    """Apply the generated matrix to a live database, or refuse and say why.

    The ordering is the contract: consent, eligibility, render, durable
    snapshot, mutate, probe. A failure at or after the mutation restores the
    snapshot before raising, so the database is never left in the half-applied
    state a failed probe implies.
    """
    consent = resolve_consent_citation(
        consent_citation, ticket=ticket, ledger_root=ledger_root
    )
    assert_matrix_appliable(matrix, phase=phase)
    rendered = render_application_database_acl_sql(matrix, phase=phase)

    snapshot = capture_snapshot(matrix)
    _write_snapshot(snapshot_path, snapshot)

    probes = plan_connection_probes(matrix)
    if not execute:
        return ModelAclApplyReport(
            mutated=False,
            probes_run=0,
            probes_passed=0,
            snapshot_path=str(snapshot_path),
            consent=consent,
            probe_descriptions=tuple(probe.describe() for probe in probes),
        )

    execute_sql(rendered)

    def _restore() -> None:
        if restore_snapshot is not None:
            restore_snapshot(snapshot_path)

    passed = 0
    for probe in probes:
        try:
            ok = run_probe(probe)
        except LookupError as error:
            _restore()
            raise AclApplyRefusalError(
                f"connection probe could not be run: {probe.describe()} "
                f"({error}); an unverifiable principal fails the run"
            ) from error
        if not ok:
            _restore()
            raise AclApplyRefusalError(
                f"connection probe failed: {probe.describe()}; "
                f"restored the pre-change ACL snapshot from {snapshot_path}"
            )
        passed += 1

    return ModelAclApplyReport(
        mutated=True,
        probes_run=len(probes),
        probes_passed=passed,
        snapshot_path=str(snapshot_path),
        consent=consent,
        probe_descriptions=tuple(probe.describe() for probe in probes),
    )


def iter_probe_env_names(probes: Iterable[ModelAclConnectionProbe]) -> tuple[str, ...]:
    """Name every environment variable a probe set needs, values excluded."""
    names = {
        resolve_role_dsn_env_name(probe.principal)
        for probe in probes
        if probe.principal is not None
    }
    return tuple(sorted(names))


__all__ = [
    "APPROVERS",
    "AclApplyRefusalError",
    "AclSnapshot",
    "apply_application_database_acl",
    "assert_matrix_appliable",
    "iter_probe_env_names",
    "plan_connection_probes",
    "resolve_consent_citation",
]
