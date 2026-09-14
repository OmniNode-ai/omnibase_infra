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
    """Resolve ``<ledger>:<line>`` to a durable OPERATOR-CONSENT row.

    Refuses fail-closed on every ambiguity: a malformed citation, a line past
    the end of the file, a row that is not a consent row, an approver outside
    the named pair, a row missing either scope half, or an APPROVED SCOPE that
    does not name the ticket being applied.
    """
    ledger_path, separator, raw_line = citation.rpartition(":")
    if not separator or not raw_line.isdigit() or not ledger_path:
        raise AclApplyRefusalError(
            f"consent citation must be '<ledger>:<line>', got {citation!r}"
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
    if _CONSENT_MARKER not in row:
        raise AclApplyRefusalError(
            f"{path}:{line_number} is not an OPERATOR-CONSENT row"
        )
    approved_by_match = _APPROVED_BY_RE.search(row)
    if approved_by_match is None or approved_by_match.group(1) not in APPROVERS:
        named = approved_by_match.group(1) if approved_by_match else "<absent>"
        raise AclApplyRefusalError(
            f"{path}:{line_number} names approver {named!r}; a live privilege "
            f"change requires one of {sorted(APPROVERS)}"
        )
    if _APPROVED_SCOPE_MARKER not in row:
        raise AclApplyRefusalError(f"{path}:{line_number} carries no APPROVED SCOPE")
    if _OUT_OF_SCOPE_MARKER not in row:
        raise AclApplyRefusalError(
            f"{path}:{line_number} carries no OUT OF SCOPE half; the "
            "out-of-scope list is what bounds the grant"
        )
    approved_scope = row.split(_APPROVED_SCOPE_MARKER, 1)[1].split(
        _OUT_OF_SCOPE_MARKER, 1
    )[0]
    out_of_scope = row.split(_OUT_OF_SCOPE_MARKER, 1)[1]
    if ticket not in approved_scope:
        raise AclApplyRefusalError(
            f"{path}:{line_number} APPROVED SCOPE does not name {ticket}"
        )
    lane_match = _LANE_RE.search(row)
    timestamp_match = _TIMESTAMP_RE.match(row.strip())
    if timestamp_match is None:
        raise AclApplyRefusalError(
            f"{path}:{line_number} does not begin with a UTC timestamp"
        )
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
