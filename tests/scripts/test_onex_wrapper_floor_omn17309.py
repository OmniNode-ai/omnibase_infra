# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The invocation-time floor in ``scripts/onex`` (OMN-17309).

Companion to ``test_onex_wrapper.py``, which pins INTERPRETER IDENTITY. This
file pins the other half: given that the right interpreter is about to run, is
the workspace it will run against one that has been *proven*?

The graded response is the whole point, and it is graded on one axis only --
whether the invocation mints durable evidence:

    below floor / no floor  ->  evidence subcommand: REFUSE, non-zero, no receipt
                            ->  ordinary subcommand: run, after ONE loud warning
    at or above floor       ->  silent, both

OMN-16932 is why the evidence half is a refusal rather than a warning. A probe
ran against a build nobody chose and produced a receipt. A wrong answer that
fails is recoverable; a wrong answer that is written down and cited later is not.

Hermetic and offline: a throwaway ``$OMNI_HOME`` with a copy of the wrapper, a
fake entrypoint that records argv, hand-built ``*.dist-info`` directories, and a
hand-written floor marker. Nothing runs ``uv``; no real venv is read.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WRAPPER_SOURCE = _REPO_ROOT / "scripts" / "onex"

_EXIT_BELOW_FLOOR = 3
_SENTINEL_OK = 41  # a status no shell failure mode produces by accident


class _Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.infra = root / "omnibase_infra"
        self.scripts = self.infra / "scripts"
        self.scripts.mkdir(parents=True)
        self.site_packages = (
            self.infra / ".venv" / "lib" / "python3.12" / "site-packages"
        )
        self.site_packages.mkdir(parents=True)
        self.venv_bin = self.infra / ".venv" / "bin"
        self.venv_bin.mkdir(parents=True)

        self.wrapper = self.scripts / "onex"
        shutil.copy2(_WRAPPER_SOURCE, self.wrapper)
        self.wrapper.chmod(0o755)

        self.floor = root / ".onex-workspace-floor.json"
        self.argv_log = root / "argv.log"

        entrypoint = self.venv_bin / "onex"
        entrypoint.write_text(
            f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> {self.argv_log}\nexit {_SENTINEL_OK}\n',
            encoding="utf-8",
        )
        entrypoint.chmod(0o755)

    def install_dist(self, name: str, version: str, commit: str | None = None) -> None:
        d = self.site_packages / f"{name}-{version}.dist-info"
        d.mkdir(parents=True, exist_ok=True)
        (d / "METADATA").write_text(
            f"Name: {name}\nVersion: {version}\n", encoding="utf-8"
        )
        if commit is not None:
            (d / "direct_url.json").write_text(
                json.dumps(
                    {
                        "url": "https://github.com/OmniNode-ai/omnimarket.git",
                        "vcs_info": {"vcs": "git", "commit_id": commit},
                    }
                ),
                encoding="utf-8",
            )

    def write_floor(
        self,
        distributions: dict[str, str] | None = None,
        omnimarket_commit: str = "",
        raw: str | None = None,
    ) -> None:
        if raw is not None:
            self.floor.write_text(raw, encoding="utf-8")
            return
        self.floor.write_text(
            json.dumps(
                {
                    "schema": "onex.workspace.floor.v1",
                    "generated_at": "2026-08-31T00:00:00Z",
                    "host": "test",
                    "omni_home": str(self.root),
                    "distributions": distributions or {},
                    "omnimarket_commit": omnimarket_commit,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["OMNI_HOME"] = str(self.root)
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"  # no stray `onex` to shadow
        return subprocess.run(
            ["bash", str(self.wrapper), *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            check=False,
        )


@pytest.fixture
def ws(tmp_path: Path) -> _Workspace:
    return _Workspace(tmp_path / "omni_home")


# --------------------------------------------------------------------------- #
# At or above floor: silent, both classes
# --------------------------------------------------------------------------- #
def test_at_floor_is_silent_and_execs(ws: _Workspace) -> None:
    ws.install_dist("omnibase_core", "0.46.9")
    ws.install_dist("omnimarket", "0.4.11", commit="a" * 40)
    ws.write_floor({"omnibase_core": "0.46.9"}, omnimarket_commit="a" * 40)

    proc = ws.run("delegate", "hello")

    assert proc.returncode == _SENTINEL_OK, proc.stderr
    assert "floor" not in proc.stderr.lower()
    assert ws.argv_log.read_text(encoding="utf-8").strip() == "delegate hello"


def test_ahead_of_floor_is_silent(ws: _Workspace) -> None:
    """AC3. Dev-tip dogfooding means the venv legitimately leads the last stamp.

    A check that warned here would warn constantly and be tuned out, which is
    how a real signal gets lost.
    """
    ws.install_dist("omnibase_core", "0.47.0")
    ws.write_floor({"omnibase_core": "0.46.9"})

    proc = ws.run("delegate", "hello")

    assert proc.returncode == _SENTINEL_OK
    assert "WARNING: this workspace is below" not in proc.stderr


def test_double_digit_patch_is_not_ordered_lexically(ws: _Workspace) -> None:
    """0.38.16 is ABOVE 0.38.9. A string compare would call it below.

    The live versions in this workspace are exactly in that range, so a lexical
    comparison would refuse every evidence command on a perfectly current venv.
    """
    ws.install_dist("omnibase_infra", "0.38.16")
    ws.write_floor({"omnibase_infra": "0.38.9"})

    assert ws.run("delegate", "x").returncode == _SENTINEL_OK


# --------------------------------------------------------------------------- #
# AC1 -- below floor
# --------------------------------------------------------------------------- #
def test_below_floor_refuses_an_evidence_subcommand(ws: _Workspace) -> None:
    ws.install_dist("omnibase_compat", "0.5.5")  # the OMN-16262 downgrade
    ws.write_floor({"omnibase_compat": "0.5.6"})

    proc = ws.run("delegate", "reply with ok")

    assert proc.returncode == _EXIT_BELOW_FLOOR
    assert "REFUSED" in proc.stderr
    assert "0.5.5" in proc.stderr and "0.5.6" in proc.stderr
    assert not ws.argv_log.exists(), "the CLI must not have run at all"


def test_below_floor_warns_but_runs_an_ordinary_subcommand(ws: _Workspace) -> None:
    ws.install_dist("omnibase_compat", "0.5.5")
    ws.write_floor({"omnibase_compat": "0.5.6"})

    proc = ws.run("info")

    assert proc.returncode == _SENTINEL_OK
    assert proc.stderr.count("WARNING: this workspace is below the proven floor.") == 1
    assert ws.argv_log.read_text(encoding="utf-8").strip() == "info"


def test_missing_distribution_reads_as_below_floor(ws: _Workspace) -> None:
    ws.install_dist("omnibase_core", "0.46.9")
    ws.write_floor({"omnibase_core": "0.46.9", "omnibase_compat": "0.5.6"})

    proc = ws.run("delegate", "x")

    assert proc.returncode == _EXIT_BELOW_FLOOR
    assert "NOT INSTALLED" in proc.stderr


def test_omnimarket_commit_mismatch_reads_as_unproven(ws: _Workspace) -> None:
    """There is no ordering on commits, so "different" is "not the proven build".

    This is the OMN-14060 comparison surfaced one layer earlier -- before the
    interpreter that would otherwise mint the receipt has even started.
    """
    ws.install_dist("omnimarket", "0.4.11", commit="b" * 40)
    ws.write_floor({}, omnimarket_commit="a" * 40)

    proc = ws.run("skill", "merge_sweep")

    assert proc.returncode == _EXIT_BELOW_FLOOR
    assert "not the proven commit" in proc.stderr


# --------------------------------------------------------------------------- #
# AC2 -- missing / unusable floor fails closed for evidence
# --------------------------------------------------------------------------- #
def test_absent_floor_refuses_evidence_and_warns_on_ordinary(ws: _Workspace) -> None:
    ws.install_dist("omnibase_core", "0.46.9")
    assert not ws.floor.exists()

    refused = ws.run("delegate", "x")
    assert refused.returncode == _EXIT_BELOW_FLOOR
    assert "no floor marker" in refused.stderr

    permitted = ws.run("info")
    assert permitted.returncode == _SENTINEL_OK


def test_empty_floor_document_is_unknown_not_a_pass(ws: _Workspace) -> None:
    ws.install_dist("omnibase_core", "0.46.9")
    ws.write_floor({}, omnimarket_commit="")

    proc = ws.run("delegate", "x")

    assert proc.returncode == _EXIT_BELOW_FLOOR
    assert "records no distributions" in proc.stderr


def test_corrupt_floor_is_unknown_not_a_pass(ws: _Workspace) -> None:
    ws.install_dist("omnibase_core", "0.46.9")
    ws.write_floor(raw="this is not json at all\n")

    assert ws.run("delegate", "x").returncode == _EXIT_BELOW_FLOOR


def test_pypi_omnimarket_with_no_direct_url_is_unknown_not_a_pass(
    ws: _Workspace,
) -> None:
    ws.install_dist("omnimarket", "0.4.10")  # no direct_url.json
    ws.write_floor({}, omnimarket_commit="a" * 40)

    proc = ws.run("delegate", "x")

    assert proc.returncode == _EXIT_BELOW_FLOOR
    assert "cannot be identified" in proc.stderr


def test_venv_with_no_distributions_is_unknown_not_a_pass(ws: _Workspace) -> None:
    ws.write_floor({"omnibase_core": "0.46.9"})

    proc = ws.run("delegate", "x")

    assert proc.returncode == _EXIT_BELOW_FLOOR
    assert "no installed distributions" in proc.stderr


# --------------------------------------------------------------------------- #
# Evidence classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "subcommand",
    [
        "delegate",
        "skill",
        "node",
        "run-node",
        "run",
        "gate",
        "occ",
        "compliance",
        "validate",
        "doctor",
        "health",
        "db",
        "ledger",
    ],
)
def test_every_declared_evidence_subcommand_refuses(
    ws: _Workspace, subcommand: str
) -> None:
    ws.install_dist("omnibase_compat", "0.5.5")
    ws.write_floor({"omnibase_compat": "0.5.6"})
    assert ws.run(subcommand, "x").returncode == _EXIT_BELOW_FLOOR


@pytest.mark.parametrize(
    "subcommand", ["info", "config", "new", "scaffold-channel-adapter"]
)
def test_ordinary_subcommands_are_permitted(ws: _Workspace, subcommand: str) -> None:
    ws.install_dist("omnibase_compat", "0.5.5")
    ws.write_floor({"omnibase_compat": "0.5.6"})
    assert ws.run(subcommand).returncode == _SENTINEL_OK


@pytest.mark.parametrize(
    "flag", ["--output", "--receipt", "--report", "--emit-receipt", "--evidence"]
)
def test_an_output_flag_makes_any_invocation_evidence_producing(
    ws: _Workspace, flag: str
) -> None:
    """An ordinary subcommand asked to WRITE its result is minting evidence.

    The file outlives the invocation and will be cited, which is the property
    that matters -- not which subcommand happened to produce it.
    """
    ws.install_dist("omnibase_compat", "0.5.5")
    ws.write_floor({"omnibase_compat": "0.5.6"})
    assert ws.run("info", flag, str(ws.root / "x.json")).returncode == _EXIT_BELOW_FLOOR
    assert (
        ws.run("info", f"{flag}={ws.root / 'x.json'}").returncode == _EXIT_BELOW_FLOOR
    )


def test_leading_global_options_do_not_hide_the_subcommand(ws: _Workspace) -> None:
    """``onex -v delegate ...`` is still a delegate."""
    ws.install_dist("omnibase_compat", "0.5.5")
    ws.write_floor({"omnibase_compat": "0.5.6"})
    assert ws.run("-v", "delegate", "x").returncode == _EXIT_BELOW_FLOOR


def test_bare_invocation_with_no_subcommand_is_ordinary(ws: _Workspace) -> None:
    """``onex --help`` must never be refused -- it is how you find out what to run."""
    ws.install_dist("omnibase_compat", "0.5.5")
    ws.write_floor({"omnibase_compat": "0.5.6"})
    assert ws.run("--help").returncode == _SENTINEL_OK


# --------------------------------------------------------------------------- #
# AC4 -- the hot path starts no interpreter and touches no network
# --------------------------------------------------------------------------- #
def test_check_works_with_no_python_and_no_uv_on_path(ws: _Workspace) -> None:
    """Run the whole wrapper with an empty PATH except coreutils and bash.

    If the floor check ever grows a ``python``/``uv``/``curl`` call, this test
    stops passing -- which is the only durable way to keep a cheap check cheap.
    """
    ws.install_dist("omnibase_compat", "0.5.5")
    ws.write_floor({"omnibase_compat": "0.5.6"})

    bin_dir = ws.root / "minimal_bin"
    bin_dir.mkdir()
    for tool in ("awk", "bash", "dirname", "cat", "readlink", "printf", "sed"):
        for candidate in ("/usr/bin", "/bin"):
            source = Path(candidate) / tool
            if source.exists():
                (bin_dir / tool).symlink_to(source)
                break

    env = dict(os.environ)
    env["OMNI_HOME"] = str(ws.root)
    env["PATH"] = str(bin_dir)
    proc = subprocess.run(
        ["bash", str(ws.wrapper), "delegate", "x"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )

    assert proc.returncode == _EXIT_BELOW_FLOOR, proc.stderr
    assert "0.5.5" in proc.stderr


def test_the_venvs_own_python_is_never_invoked(ws: _Workspace) -> None:
    """A broken interpreter must not make the check unable to answer.

    A venv mid-``uv sync`` is exactly when the floor matters most, so the check
    reads directory names rather than asking the environment about itself.
    """
    broken = ws.venv_bin / "python"
    broken.write_text("#!/usr/bin/env bash\nexit 70\n", encoding="utf-8")
    broken.chmod(0o755)
    ws.install_dist("omnibase_compat", "0.5.5")
    ws.write_floor({"omnibase_compat": "0.5.6"})

    assert ws.run("delegate", "x").returncode == _EXIT_BELOW_FLOOR


# --------------------------------------------------------------------------- #
# AC6 -- no bypass
# --------------------------------------------------------------------------- #
def test_no_env_var_bypasses_the_refusal(ws: _Workspace) -> None:
    """The existing override is scoped to the missing-entrypoint self-heal only.

    Setting it must not become a way to mint a receipt from an unproven build.
    """
    ws.install_dist("omnibase_compat", "0.5.5")
    ws.write_floor({"omnibase_compat": "0.5.6"})

    env = dict(os.environ)
    env["OMNI_HOME"] = str(ws.root)
    env["ONEX_WRAPPER_NO_RECONCILE"] = "1"
    proc = subprocess.run(
        ["bash", str(ws.wrapper), "delegate", "x"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == _EXIT_BELOW_FLOOR


def test_wrapper_declares_no_floor_bypass_variable() -> None:
    """Structural: the wrapper source must not grow one later.

    A bypass is the single change that would silently undo this whole ticket, so
    it is asserted against the source rather than left to review vigilance.
    """
    source = _WRAPPER_SOURCE.read_text(encoding="utf-8")
    for forbidden in (
        "ONEX_ALLOW_BELOW_FLOOR",
        "ONEX_SKIP_FLOOR",
        "ONEX_NO_FLOOR",
        "ONEX_FLOOR_BYPASS",
    ):
        assert forbidden not in source
