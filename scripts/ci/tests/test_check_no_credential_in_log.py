# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for check_no_credential_in_log.py (OMN-17423).

The gate's own self-test asserts a COUNT against the committed fixture. These
tests assert the RULES, so a change that keeps the count while inverting a
decision still fails. The precision cases are not decoration: the first draft
of this gate flagged seven token-accounting metrics in ``src/``, and a gate
that cries wolf is a gate someone deletes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS_CI = Path(__file__).parent.parent
FIXTURE = SCRIPTS_CI / "tests" / "fixtures" / "credential_in_log_fixture.py"


def _load_gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_no_credential_in_log", SCRIPTS_CI / "check_no_credential_in_log.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


@pytest.mark.unit
class TestNameMatching:
    @pytest.mark.parametrize(
        "name",
        [
            "api_key",
            "apiKey",
            "plaintext_key",
            "access_token",
            "refresh_token",
            "session_token",
            "gateway_token",
            "client_secret",
            "password",
            "linear_api_key",
        ],
    )
    def test_credential_names_are_caught(self, name: str) -> None:
        assert gate._credential_hit(name) is not None

    @pytest.mark.parametrize(
        "name",
        [
            "api_key_count",
            "api_key_id",
            "api_key_ref",
            "key_hash",
            "token_count",
            "token_savings_pct",
            "total_direct_tokens",
            "tokens_total_raw",
            "_token_threshold",
            "secrets_seeded",
            "_SECRET_POLICY_ENV_VAR",
        ],
    )
    def test_reference_and_metric_names_are_not_caught(self, name: str) -> None:
        """Every one of these is live in ``src/`` today."""
        assert gate._credential_hit(name) is None


@pytest.mark.unit
class TestFormatStringMatching:
    @pytest.mark.parametrize(
        "text",
        [
            "api_key=%s",
            "minted access_token: %s",
            "gateway_token: %s",
            'client_secret="%s"',
            "API_KEY=%s",
            "token=%s",
        ],
    )
    def test_assigned_credentials_are_caught(self, text: str) -> None:
        assert gate._format_string_hit(text) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "LINEAR_API_KEY is not set - cannot call Linear API.",
            "Infisical needs INFISICAL_CLIENT_ID and INFISICAL_CLIENT_SECRET",
            "refresh token expired, re-minting",
            "api_key_id=%s",
            "tenant acme resolved in 12ms",
        ],
    )
    def test_env_var_names_and_prose_are_not_caught(self, text: str) -> None:
        assert gate._format_string_hit(text) is None

    def test_env_var_name_assigned_a_value_is_still_caught(self) -> None:
        """An ALL-CAPS name is suppressed as a NAME, never as a value."""
        assert gate._format_string_hit("LINEAR_API_KEY=%s") is not None


@pytest.mark.unit
class TestFixtureAndSelfTest:
    def test_fixture_yields_the_expected_count(self) -> None:
        assert len(gate._check_file(FIXTURE)) == 9

    def test_self_test_passes_at_the_committed_count(self) -> None:
        assert gate.run_self_test(FIXTURE, 9) == 0

    def test_self_test_fails_on_a_stale_count(self) -> None:
        """The self-test is itself non-vacuous."""
        assert gate.run_self_test(FIXTURE, 8) == 1

    def test_missing_fixture_fails_closed(self) -> None:
        assert gate.run_self_test(FIXTURE.parent / "nope.py", 9) == 1


@pytest.mark.unit
class TestScan:
    def test_shared_runtime_path_is_clean(self) -> None:
        """AC4's live claim: the path AC2 covers has no credential log site."""
        repo_root = SCRIPTS_CI.parent.parent
        assert gate.run_normal(repo_root / "src" / "omnibase_infra") == 0

    def test_missing_root_fails_closed(self, tmp_path: Path) -> None:
        assert gate.run_normal(tmp_path / "absent") == 1

    def test_fixture_directories_are_excluded(self) -> None:
        """The fixture must not be able to redden the normal scan."""
        collected = gate._collect_py_files(SCRIPTS_CI / "tests")
        assert FIXTURE not in collected

    def test_suppression_requires_the_annotation(self, tmp_path: Path) -> None:
        source = 'import logging\nlogger = logging.getLogger(__name__)\nlogger.info("api_key=%s", v)\n'
        unsuppressed = tmp_path / "a.py"
        unsuppressed.write_text(source, encoding="utf-8")
        assert len(gate._check_file(unsuppressed)) == 1

        suppressed = tmp_path / "b.py"
        suppressed.write_text(
            source.replace(
                '", v)', '", v)  # credential-log-allow: test of the escape hatch'
            ),
            encoding="utf-8",
        )
        assert gate._check_file(suppressed) == []
