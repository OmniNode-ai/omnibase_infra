# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
import re
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
from _pytest.pytester import Pytester

from omnibase_infra.testing.rsd_postgres_acceptance_capability import (
    CapabilityResolver,
    ModelRsdPostgresAcceptanceEvidence,
    RsdPostgresAcceptanceCapability,
    RsdPostgresAcceptanceResolutionError,
    resolve_postgres_lifecycle_factory,
)
from omnibase_infra.testing.rsd_postgres_acceptance_plugin import (
    load_overlay,
    postgres_lifecycle_connection_factory,
)

pytest_plugins = ("pytester",)

CAPABILITY_UUID = "123e4567-e89b-42d3-a456-426614174000"


def _evidence(
    capability_ref: str = "capability://rsd/postgres/acceptance",
) -> ModelRsdPostgresAcceptanceEvidence:
    return ModelRsdPostgresAcceptanceEvidence(
        schema_version="rsd_postgres_acceptance_evidence.v1",
        capability_ref=capability_ref,
        target_identity_attestation_ref=(
            "attestation://rsd/postgres/target/123e4567-e89b-42d3-a456-426614174000"
        ),
        session_identity_attestation_ref=(
            "attestation://rsd/postgres/session/223e4567-e89b-42d3-a456-426614174000"
        ),
        role_identity_attestation_ref=(
            "attestation://rsd/postgres/role/323e4567-e89b-42d3-a456-426614174000"
        ),
        authority_disposition="trusted_operator_factory_contract",
        fresh_lease=True,
        exclusive_lease=True,
        transaction_idle=True,
    )


def test_overlay_is_typed_and_topology_free() -> None:
    path = (
        Path(__file__).parents[3]
        / "docker/lane-overlays/dev.rsd-postgres-acceptance.yaml"
    )
    overlay = load_overlay(path)
    assert overlay.lane == "dev" and overlay.locale == "lab"
    overlay_text = path.read_text()
    forbidden = re.compile(
        r"(?i)(?:\b(?:host|port|dsn|password|passwd|username|user|url|endpoint|ip|address|env)\s*:|"
        r"\b(?:host|port|dsn|password|passwd|username|user|url|endpoint|ip|address|env)\s*=|"
        r"https?://|(?:\d{1,3}\.){3}\d{1,3}|(?:postgres(?:ql)?|mysql)://)"
    )
    assert forbidden.search(overlay_text) is None


def test_overlay_allows_opaque_uuid_capability_reference(tmp_path: Path) -> None:
    path = tmp_path / "uuid-overlay.yaml"
    path.write_text(
        "schema_version: rsd_postgres_acceptance_overlay.v1\n"
        "lane: dev\n"
        "locale: lab\n"
        "rsd_distribution_ref: omninode-rsd/0.1.0\n"
        "postgres_capability_ref: capability://rsd/postgres/"
        f"{CAPABILITY_UUID}\n"
    )
    overlay = load_overlay(path)
    assert overlay.postgres_capability_ref.endswith(CAPABILITY_UUID)


def test_operator_capability_factory_is_fresh_exclusive_and_idle() -> None:
    """Exercise the injected factory contract with an offline fake only."""
    active_connections = 0

    class FakeConnection:
        def __init__(self) -> None:
            self.transaction_id = uuid4()
            self.transaction_state = "IDLE"
            self.exclusive = True
            self.closed = False

    def factory() -> AbstractContextManager[FakeConnection]:
        @contextmanager
        def connection() -> Iterator[FakeConnection]:
            nonlocal active_connections
            if active_connections:
                raise AssertionError("connection CMs must be exclusive")
            active_connections += 1
            value = FakeConnection()
            try:
                yield value
            finally:
                assert value.transaction_state == "IDLE"
                value.closed = True
                active_connections -= 1

        return connection()

    first_cm = factory()
    with first_cm as first:
        assert first.exclusive
        assert first.transaction_state == "IDLE"
        with pytest.raises(AssertionError, match="exclusive"):
            with factory():
                pass
    second_cm = factory()
    assert first_cm is not second_cm
    with second_cm as second:
        assert second.exclusive
        assert second.transaction_state == "IDLE"
        assert first.transaction_id != second.transaction_id
        assert first.closed
    assert second.closed


def test_overlay_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: rsd_postgres_acceptance_overlay.v1\nlane: dev\nlocale: lab\nrsd_distribution_ref: omninode-rsd/0.1.0\npostgres_capability_ref: capability://rsd/postgres/acceptance\nhost: bad\n"
    )
    with pytest.raises(Exception):
        load_overlay(path)


def test_capability_resolution_passes_opaque_ref_to_operator() -> None:
    capability_ref = f"capability://rsd/postgres/{CAPABILITY_UUID}"
    evidence = _evidence(capability_ref)
    calls: list[str] = []

    def factory() -> AbstractContextManager[object]:
        raise AssertionError("factory must not be called during resolution")

    def resolver(ref: str) -> RsdPostgresAcceptanceCapability:
        calls.append(ref)
        return RsdPostgresAcceptanceCapability(factory, evidence)

    assert resolve_postgres_lifecycle_factory(resolver, capability_ref) is factory
    assert calls == [capability_ref]


@pytest.mark.parametrize(
    "hostile_ref",
    [
        "capability://rsd/postgres/db.example",
        "capability://rsd/postgres/../target",
        "capability://rsd/postgres/192.168.1.10",
        "capability://rsd/postgres/123e4567-e89b-12d3-a456-426614174000",
        "capability://rsd/postgres/123e4567-e89b-42d3-a456-426614174000/extra",
    ],
)
def test_capability_resolution_rejects_topology_bearing_refs(
    hostile_ref: str,
) -> None:
    with pytest.raises(
        RsdPostgresAcceptanceResolutionError, match=r"invalid.*reference"
    ):
        resolve_postgres_lifecycle_factory(
            lambda _: pytest.fail("resolver must not receive hostile ref"), hostile_ref
        )


def test_capability_resolution_rejects_missing_identity_evidence() -> None:
    def factory() -> AbstractContextManager[object]:
        raise AssertionError("factory must not be called")

    class MissingEvidence:
        connection_factory = factory

    with pytest.raises(
        RsdPostgresAcceptanceResolutionError, match="typed identity evidence"
    ):
        resolve_postgres_lifecycle_factory(
            cast("CapabilityResolver", lambda _: MissingEvidence()),
            "capability://rsd/postgres/acceptance",
        )


def test_capability_resolution_normalizes_exploding_properties() -> None:
    class ExplodingCapability:
        @property
        def connection_factory(self) -> object:
            raise RuntimeError("postgres://topology-secret")

        @property
        def evidence(self) -> object:
            raise RuntimeError("role=topology-secret")

    with pytest.raises(
        RsdPostgresAcceptanceResolutionError,
        match="operator capability inspection failed",
    ) as failure:
        resolve_postgres_lifecycle_factory(
            cast("CapabilityResolver", lambda _: ExplodingCapability()),
            "capability://rsd/postgres/acceptance",
        )
    assert "topology-secret" not in str(failure.value)
    assert failure.value.__cause__ is None


def test_evidence_json_round_trip_and_strict_disposition() -> None:
    evidence = _evidence()
    restored = ModelRsdPostgresAcceptanceEvidence.model_validate_json(
        evidence.model_dump_json()
    )
    assert restored == evidence
    with pytest.raises(Exception):
        ModelRsdPostgresAcceptanceEvidence.model_validate(
            evidence.model_dump() | {"authority_disposition": "operator_claimed"}
        )


@pytest.mark.parametrize(
    ("field", "hostile_value"),
    [
        (
            "target_identity_attestation_ref",
            "attestation://rsd/postgres/target/db.example",
        ),
        (
            "session_identity_attestation_ref",
            "attestation://rsd/postgres/session/../session",
        ),
        (
            "role_identity_attestation_ref",
            "attestation://rsd/postgres/role/192.168.1.10",
        ),
        (
            "target_identity_attestation_ref",
            "attestation://rsd/postgres/target/pg.internal",
        ),
    ],
)
def test_evidence_rejects_topology_bearing_attestation_refs(
    field: str, hostile_value: str
) -> None:
    with pytest.raises(Exception):
        ModelRsdPostgresAcceptanceEvidence.model_validate(
            _evidence().model_dump() | {field: hostile_value}
        )


def test_explicit_plugin_harness_covers_missing_and_mismatch_paths(
    pytester: Pytester,
) -> None:
    """Exercise the actual ``-p`` entrypoint with an isolated fake operator."""
    pytester.makeconftest(
        """
        from contextlib import contextmanager
        from omnibase_infra.testing.rsd_postgres_acceptance_capability import (
            ModelRsdPostgresAcceptanceEvidence,
            RsdPostgresAcceptanceCapability,
        )

        EXPECTED_REF = "capability://rsd/postgres/acceptance"

        def _evidence():
            return ModelRsdPostgresAcceptanceEvidence(
                schema_version="rsd_postgres_acceptance_evidence.v1",
                capability_ref=EXPECTED_REF,
                target_identity_attestation_ref=(
                    "attestation://rsd/postgres/target/123e4567-e89b-42d3-a456-426614174000"
                ),
                session_identity_attestation_ref=(
                    "attestation://rsd/postgres/session/223e4567-e89b-42d3-a456-426614174000"
                ),
                role_identity_attestation_ref=(
                    "attestation://rsd/postgres/role/323e4567-e89b-42d3-a456-426614174000"
                ),
                authority_disposition="trusted_operator_factory_contract",
                fresh_lease=True,
                exclusive_lease=True,
                transaction_idle=True,
            )

        @contextmanager
        def _connection():
            yield object()

        def _factory():
            return _connection()

        def _resolver(ref):
            if ref != EXPECTED_REF:
                raise LookupError("postgres://topology-secret")
            return RsdPostgresAcceptanceCapability(_factory, _evidence())

        def pytest_configure(config):
            mode = config.getoption("--fake-capability-mode")
            if mode == "valid":
                config.rsd_postgres_acceptance_capability_resolver = _resolver
            elif mode == "missing-evidence":
                config.rsd_postgres_acceptance_capability_resolver = (
                    lambda ref: RsdPostgresAcceptanceCapability(_factory, None)
                )
            elif mode == "explode-factory":
                class ExplodingFactoryCapability:
                    @property
                    def connection_factory(self):
                        raise RuntimeError("postgres://topology-secret")

                    evidence = _evidence()

                config.rsd_postgres_acceptance_capability_resolver = (
                    lambda ref: ExplodingFactoryCapability()
                )
            elif mode == "explode-evidence":
                class ExplodingEvidenceCapability:
                    connection_factory = _factory

                    @property
                    def evidence(self):
                        raise RuntimeError("role=topology-secret")

                config.rsd_postgres_acceptance_capability_resolver = (
                    lambda ref: ExplodingEvidenceCapability()
                )

        def pytest_addoption(parser):
            parser.addoption("--fake-capability-mode", default="valid")
        """
    )
    pytester.makepyfile(
        test_plugin_harness="""
        def test_factory_is_injected(postgres_lifecycle_connection_factory):
            with postgres_lifecycle_connection_factory() as connection:
                assert connection is not None
        """
    )
    overlay = pytester.path / "overlay.yaml"
    overlay.write_text(
        "schema_version: rsd_postgres_acceptance_overlay.v1\n"
        "lane: dev\nlocale: lab\n"
        "rsd_distribution_ref: omninode-rsd/0.1.0\n"
        "postgres_capability_ref: capability://rsd/postgres/acceptance\n"
    )
    result = pytester.runpytest(
        "-p",
        "omnibase_infra.testing.rsd_postgres_acceptance_plugin",
        "--fake-capability-mode=valid",
        f"--rsd-postgres-acceptance-overlay={overlay}",
    )
    result.assert_outcomes(passed=1)

    missing = pytester.runpytest(
        "-p", "omnibase_infra.testing.rsd_postgres_acceptance_plugin"
    )
    missing.assert_outcomes(errors=1)
    assert (
        "overlay path must be explicit" in missing.stdout.str() + missing.stderr.str()
    )

    mismatch = pytester.makefile(
        ".yaml",
        mismatch=(
            "schema_version: rsd_postgres_acceptance_overlay.v1\n"
            "lane: dev\nlocale: lab\n"
            "rsd_distribution_ref: omninode-rsd/0.1.0\n"
            "postgres_capability_ref: capability://rsd/postgres/"
            "123e4567-e89b-42d3-a456-426614174000\n"
        ),
    )
    mismatch_result = pytester.runpytest(
        "-p",
        "omnibase_infra.testing.rsd_postgres_acceptance_plugin",
        "--fake-capability-mode=valid",
        f"--rsd-postgres-acceptance-overlay={mismatch}",
    )
    mismatch_result.assert_outcomes(errors=1)
    assert "operator capability lookup failed" in (
        mismatch_result.stdout.str() + mismatch_result.stderr.str()
    )
    assert "topology-secret" not in (
        mismatch_result.stdout.str() + mismatch_result.stderr.str()
    )

    missing_resolver = pytester.runpytest(
        "-p",
        "omnibase_infra.testing.rsd_postgres_acceptance_plugin",
        "--fake-capability-mode=missing-resolver",
        f"--rsd-postgres-acceptance-overlay={overlay}",
    )
    missing_resolver.assert_outcomes(errors=1)
    assert "resolver is not injected" in (
        missing_resolver.stdout.str() + missing_resolver.stderr.str()
    )

    missing_evidence = pytester.runpytest(
        "-p",
        "omnibase_infra.testing.rsd_postgres_acceptance_plugin",
        "--fake-capability-mode=missing-evidence",
        f"--rsd-postgres-acceptance-overlay={overlay}",
    )
    missing_evidence.assert_outcomes(errors=1)
    assert "typed identity evidence" in (
        missing_evidence.stdout.str() + missing_evidence.stderr.str()
    )

    for mode in ("explode-factory", "explode-evidence"):
        exploding = pytester.runpytest(
            "-p",
            "omnibase_infra.testing.rsd_postgres_acceptance_plugin",
            f"--fake-capability-mode={mode}",
            f"--rsd-postgres-acceptance-overlay={overlay}",
        )
        exploding.assert_outcomes(errors=1)
        output = exploding.stdout.str() + exploding.stderr.str()
        assert "operator capability inspection failed" in output
        assert "topology-secret" not in output


def test_plugin_rejects_missing_resolver(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "schema_version: rsd_postgres_acceptance_overlay.v1\n"
        "lane: dev\nlocale: lab\n"
        "rsd_distribution_ref: omninode-rsd/0.1.0\n"
        "postgres_capability_ref: capability://rsd/postgres/acceptance\n"
    )

    class Config:
        def getoption(self, name: str) -> str:
            assert name == "--rsd-postgres-acceptance-overlay"
            return str(overlay)

    class Request:
        config = Config()

    fixture_function = cast(
        "Callable[[object], object]",
        postgres_lifecycle_connection_factory.__wrapped__,  # type: ignore[attr-defined]
    )
    with pytest.raises(pytest.fail.Exception, match="resolver is not injected"):
        fixture_function(Request())
