# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for canonical unified validate_no_env_fallbacks.py (OMN-10741)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate_no_env_fallbacks import run, scan_python_file, scan_shell_file


def _py(tmp_path: Path, content: str, name: str = "test_mod.py") -> Path:
    f = tmp_path / "src" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content)
    return f


def _sh(tmp_path: Path, content: str, name: str = "test_script.sh") -> Path:
    f = tmp_path / "scripts" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content)
    return f


@pytest.mark.unit
class TestPythonPatternOsEnvironGet:
    def test_detects_os_environ_get_with_localhost(self, tmp_path: Path) -> None:
        p = _py(tmp_path, 'host = os.environ.get("HOST", "localhost")\n')
        viols = scan_python_file(p)
        assert len(viols) == 1

    def test_detects_os_environ_get_with_http_localhost(self, tmp_path: Path) -> None:
        p = _py(tmp_path, 'url = os.environ.get("URL", "http://localhost:8080")\n')
        viols = scan_python_file(p)
        assert len(viols) == 1

    def test_clean_os_environ_raises(self, tmp_path: Path) -> None:
        p = _py(tmp_path, 'host = os.environ["HOST"]\n')
        assert scan_python_file(p) == []

    def test_fallback_ok_exempts_line(self, tmp_path: Path) -> None:
        p = _py(
            tmp_path,
            'host = os.environ.get("HOST", "localhost")  # fallback-ok: test harness\n',
        )
        assert scan_python_file(p) == []


@pytest.mark.unit
class TestPythonPatternOsGetenv:
    def test_detects_os_getenv_with_localhost(self, tmp_path: Path) -> None:
        p = _py(tmp_path, 'db = os.getenv("DB_HOST", "localhost")\n')
        viols = scan_python_file(p)
        assert len(viols) == 1

    def test_detects_os_getenv_with_redis_localhost(self, tmp_path: Path) -> None:
        p = _py(tmp_path, 'url = os.getenv("REDIS_URL", "redis://localhost:6379")\n')
        viols = scan_python_file(p)
        assert len(viols) == 1

    def test_clean_os_getenv_no_default(self, tmp_path: Path) -> None:
        p = _py(tmp_path, 'url = os.getenv("REDIS_URL")\n')
        assert scan_python_file(p) == []

    def test_cloud_bus_ok_exempts(self, tmp_path: Path) -> None:
        p = _py(
            tmp_path,
            'db = os.getenv("DB_HOST", "localhost")  # cloud-bus-ok\n',
        )
        assert scan_python_file(p) == []


@pytest.mark.unit
class TestPythonPatternDefaultEquals:
    def test_detects_pydantic_field_default_localhost(self, tmp_path: Path) -> None:
        p = _py(tmp_path, '    host: str = Field(default="localhost")\n')
        viols = scan_python_file(p)
        assert len(viols) == 1

    def test_detects_function_param_default_localhost(self, tmp_path: Path) -> None:
        p = _py(tmp_path, 'def connect(host: str = "localhost") -> None: ...\n')
        viols = scan_python_file(p)
        assert len(viols) == 1

    def test_clean_field_no_default(self, tmp_path: Path) -> None:
        p = _py(tmp_path, "    host: str\n")
        assert scan_python_file(p) == []


@pytest.mark.unit
class TestPythonPatternPrivateIpDefault:
    def test_detects_os_getenv_private_ip(self, tmp_path: Path) -> None:
        # Use a non-KAFKA_ var name so the kafka-no-hardcoded-fallback hook
        # doesn't also match this test data string.
        p = _py(tmp_path, 'host = os.getenv("INFRA_HOST", "192.168.86.201")\n')
        viols = scan_python_file(p)
        assert len(viols) == 1

    def test_detects_default_equals_private_ip(self, tmp_path: Path) -> None:
        p = _py(tmp_path, '    host: str = Field(default="192.168.1.100")\n')
        viols = scan_python_file(p)
        assert len(viols) == 1

    def test_fallback_ok_exempts_private_ip(self, tmp_path: Path) -> None:
        p = _py(
            tmp_path,
            'host = os.getenv("INFRA_HOST", "192.168.86.201")  # fallback-ok: lab-only\n',
        )
        assert scan_python_file(p) == []


@pytest.mark.unit
class TestPythonPatternBootstrapServers:
    def test_detects_bootstrap_servers_localhost(self, tmp_path: Path) -> None:
        p = _py(
            tmp_path, 'producer = KafkaProducer(bootstrap_servers="localhost:9092")\n'
        )
        viols = scan_python_file(p)
        assert len(viols) == 1

    def test_clean_bootstrap_servers_from_env(self, tmp_path: Path) -> None:
        p = _py(
            tmp_path,
            'producer = KafkaProducer(bootstrap_servers=os.environ["KAFKA_BOOTSTRAP_SERVERS"])\n',
        )
        assert scan_python_file(p) == []


@pytest.mark.unit
class TestDocstringSkipping:
    def test_skips_localhost_in_docstring(self, tmp_path: Path) -> None:
        content = '"""\nExample: os.getenv("HOST", "localhost")\n"""\n'
        p = _py(tmp_path, content)
        assert scan_python_file(p) == []

    def test_skips_localhost_in_same_line_docstring(self, tmp_path: Path) -> None:
        content = '"""Example: os.getenv("HOST", "localhost")"""\n'
        p = _py(tmp_path, content)
        assert scan_python_file(p) == []

    def test_skips_pure_comment_lines(self, tmp_path: Path) -> None:
        p = _py(tmp_path, '# host = os.environ.get("HOST", "localhost")\n')
        assert scan_python_file(p) == []

    def test_detects_violation_after_docstring(self, tmp_path: Path) -> None:
        content = (
            '"""Module docstring."""\n\nhost = os.environ.get("HOST", "localhost")\n'
        )
        p = _py(tmp_path, content)
        viols = scan_python_file(p)
        assert len(viols) == 1
        assert viols[0][0] == 3

    def test_embedded_triple_quotes_do_not_start_docstring(
        self, tmp_path: Path
    ) -> None:
        content = (
            'marker = """not a docstring opener"""\nurl = os.getenv("X", "localhost")\n'
        )
        p = _py(tmp_path, content)
        assert scan_python_file(p) == [(2, 'url = os.getenv("X", "localhost")')]

    def test_embedded_triple_quotes_on_fallback_line_still_scans(
        self, tmp_path: Path
    ) -> None:
        content = 'url = os.getenv("X", "localhost") + """suffix"""\n'
        p = _py(tmp_path, content)
        assert scan_python_file(p) == [
            (1, 'url = os.getenv("X", "localhost") + """suffix"""')
        ]

    def test_line_start_triple_quote_with_trailing_fallback_still_scans(
        self, tmp_path: Path
    ) -> None:
        content = '"""prefix"""; url = os.getenv("X", "localhost")\n'
        p = _py(tmp_path, content)
        assert scan_python_file(p) == [
            (1, '"""prefix"""; url = os.getenv("X", "localhost")')
        ]


@pytest.mark.unit
class TestShellPatternBashVarDefault:
    def test_detects_bash_var_localhost_default(self, tmp_path: Path) -> None:
        s = _sh(tmp_path, 'HOST="${HOST:-localhost}"\n')
        viols = scan_shell_file(s)
        assert len(viols) == 1

    def test_detects_bash_var_http_localhost_default(self, tmp_path: Path) -> None:
        s = _sh(tmp_path, 'URL="${API_URL:-http://localhost:8080}"\n')
        viols = scan_shell_file(s)
        assert len(viols) == 1

    def test_detects_bash_private_ip_default(self, tmp_path: Path) -> None:
        s = _sh(tmp_path, 'HOST="${KAFKA_HOST:-192.168.86.201}"\n')
        viols = scan_shell_file(s)
        assert len(viols) == 1

    def test_clean_bash_var_no_default(self, tmp_path: Path) -> None:
        s = _sh(tmp_path, 'HOST="${HOST}"\n')
        assert scan_shell_file(s) == []

    def test_fallback_ok_exempts_shell(self, tmp_path: Path) -> None:
        s = _sh(tmp_path, 'HOST="${HOST:-localhost}"  # fallback-ok: bind address\n')
        assert scan_shell_file(s) == []

    def test_skips_comment_lines_in_shell(self, tmp_path: Path) -> None:
        s = _sh(tmp_path, '# HOST="${HOST:-localhost}"\n')
        assert scan_shell_file(s) == []


@pytest.mark.unit
class TestRunFunction:
    def test_run_finds_violations_across_src_and_scripts(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "src" / "config.py").write_text(
            'url = os.environ.get("DB_URL", "postgresql://localhost/db")\n'
        )
        (tmp_path / "scripts" / "setup.sh").write_text('DB="${DB_URL:-localhost}"\n')

        violations = run(
            scan_roots=[tmp_path / "src", tmp_path / "scripts"],
            repo_root=tmp_path,
        )
        assert len(violations) == 2

    def test_run_skips_tests_directory(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "tests").mkdir()
        (tmp_path / "src" / "tests" / "conftest.py").write_text(
            'url = os.environ.get("DB_URL", "localhost")\n'
        )

        violations = run(
            scan_roots=[tmp_path / "src"],
            repo_root=tmp_path,
        )
        assert violations == []

    def test_run_skips_self(self, tmp_path: Path) -> None:
        (tmp_path / "scripts").mkdir()
        script = tmp_path / "scripts" / "validate_no_env_fallbacks.py"
        script.write_text('# pattern references: os.environ.get("X", "localhost")\n')

        violations = run(
            scan_roots=[tmp_path / "scripts"],
            repo_root=tmp_path,
        )
        assert violations == []

    def test_run_returns_empty_when_clean(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "settings.py").write_text('HOST = os.environ["HOST"]\n')

        violations = run(
            scan_roots=[tmp_path / "src"],
            repo_root=tmp_path,
        )
        assert violations == []

    def test_run_violation_tuple_format(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text(
            'host = os.getenv("HOST", "localhost")\n'
        )

        violations = run(
            scan_roots=[tmp_path / "src"],
            repo_root=tmp_path,
        )
        assert len(violations) == 1
        filepath, lineno, line_text = violations[0]
        assert "app.py" in filepath
        assert lineno == 1
        assert "localhost" in line_text
