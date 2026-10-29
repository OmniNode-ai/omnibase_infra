# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lab proof reads the omnimarket package where omnimarket runs (OMN-21217).

omnibase_infra's runtime image installs no omnimarket (OMN-21279), so a proof of
an omnimarket change that read the package only in the two infra kernels found
0 of 9,350 files (omnimarket#4540) and never saw a changed node log.
The proof now builds the composed omnimarket image, boots the ``local-market``
bundle on it, and reads every container that runs it, the names coming from the
render and not from a list in the script. These tests drive the host's own shell
functions against a fake ``docker`` and ``uv`` and the pool judge against what
they print. A container that has no omnimarket must still fail the proof.
"""

from __future__ import annotations

import importlib.util
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

_HERE = Path(__file__)
_BASE = _HERE.with_name("test_prepr_pool_pr_head_receipt_omn19566.py")
_POOL_TESTS = _HERE.with_name("test_prepr_runtime_pool_omn18893.py")


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load("prepr_pool_receipt_helpers_omn21217", _BASE)
fakes = _load("prepr_pool_fakes_omn21217", _POOL_TESTS)
pool = base.pool
PROVE_SH = base.PROVE_SH

PROJECT = "omnibase-infra-local"
KERNEL_M = f"{PROJECT}-omninode-runtime"
KERNEL_E = f"{PROJECT}-runtime-effects"
API = f"{PROJECT}-projection-api"
WRITER = f"{PROJECT}-omnimarket-projection-delegation"
COMPOSED = [KERNEL_M, KERNEL_E, API, WRITER]


def _heredoc(name: str) -> str:
    text = PROVE_SH.read_text(encoding="utf-8")
    start = text.index(f"IFS= read -r -d '' {name} <<'PY' || :\n")
    body = text[start:].split("\n", 1)[1]
    return body[: body.index("\nPY\n") + 1]


def _functions(*names: str) -> str:
    return "".join(base._shell_function(n) for n in names)


def _write_exe(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class Host:
    """A fake docker host: which containers carry which package, and their logs."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.site = tmp_path / "site"
        self.site.mkdir()
        self.logs = tmp_path / "logs"
        self.logs.mkdir()
        self.calls = tmp_path / "calls.txt"
        self.calls.write_text("", encoding="utf-8")
        self.names = tmp_path / "names.txt"
        self.names.write_text("\n".join(COMPOSED) + "\n", encoding="utf-8")
        self.carriers: list[str] = []
        self.work = tmp_path / "work"
        (self.work / "root").mkdir(parents=True)
        # docker: exec runs the package listing only where the package is
        # installed; logs reads a canned file; every call is recorded
        _write_exe(
            self.bin / "docker",
            f"""#!/bin/bash
echo "docker $*" >> {self.calls}
case "$1" in
  exec)
    c=$2; shift 2
    case " $FAKE_CARRIERS " in
      *" $c "*) PYTHONPATH={self.site} exec python3 "${{@:2}}" ;;
      *) echo "ModuleNotFoundError: No module named '$4'" >&2; exit 1 ;;
    esac ;;
  logs) cat "{self.logs}/$2" 2>/dev/null ;;
  build) exit 0 ;;
esac
""",
        )
        _write_exe(
            self.bin / "uv",
            f"""#!/bin/bash
echo "uv $*" >> {self.calls}
cat {self.names}
""",
        )

    def package(self, repo: str, files: dict[str, str]) -> Path:
        """A git build tree of ``repo`` and the same package installed in site."""
        tree = self.work / "root" / repo
        for rel, body in files.items():
            for dest in (tree / "src" / repo / rel, self.site / repo / rel):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(body, encoding="utf-8")
        for args in (
            ["init", "-q", "-b", "dev"],
            ["add", "-A"],
            ["-c", "user.name=t", "-c", "user.email=t@t.invalid", "commit", "-qm", "t"],
            ["update-ref", "refs/remotes/origin/dev", "HEAD"],
        ):
            subprocess.run(
                ["git", "-C", str(tree), *args],
                check=True,
                capture_output=True,
                env=scrub_git_location_env(),
            )
        return tree

    def run(self, body: str, **env: str) -> subprocess.CompletedProcess[str]:
        script = (
            "R=$W/root\n"
            f"M={KERNEL_M}; E={KERNEL_E}; P={PROJECT}\n"
            f"IMG_LIST_PY={_quote(_heredoc('IMG_LIST_PY'))}\n"
            f"PKG_TREE_PY={_quote(_heredoc('PKG_TREE_PY'))}\n" + body
        )
        full = {
            "PATH": f"{self.bin}:/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "W": str(self.work),
            "TAG": f"t{abs(hash(str(self.tmp)))}",
            "FAKE_CARRIERS": " ".join(self.carriers),
            "HOME": str(self.tmp),
            **env,
        }
        return subprocess.run(
            ["bash", "-c", script],
            env=full,
            capture_output=True,
            text=True,
            check=False,
        )


def _quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


MARKET_FILES = {"__init__.py": "V = 1\n", "nodes/n.py": "X = 2\n"}


def _identity(host: Host, **env: str) -> subprocess.CompletedProcess[str]:
    return host.run(
        _functions("market_under_test", "proof_containers", "identity_readback")
        + "identity_readback\n",
        **env,
    )


def test_an_omnimarket_proof_reads_the_package_in_every_container_that_runs_it(
    tmp_path: Path,
) -> None:
    host = Host(tmp_path)
    host.package("omnimarket", MARKET_FILES)
    (host.work / "composed-containers").write_text("\n".join(COMPOSED) + "\n")
    host.carriers = COMPOSED
    out = _identity(host, MARKET_PR="4540")
    assert out.returncode == 0, out.stderr
    assert "identity-subject omnimarket" in out.stdout
    for c in COMPOSED:
        assert f"pkg-tree omnimarket {c} " in out.stdout, out.stdout
    assert out.stdout.count("match=yes") == len(COMPOSED), out.stdout
    probe = fakes.GOOD["probe"].split("\n", 1)[1]
    judged = pool.judge({**fakes.GOOD, "probe": out.stdout + probe})
    assert judged.checks["image_identity"] is True, judged


@pytest.mark.parametrize("missing", COMPOSED)
def test_one_container_with_no_omnimarket_still_fails_the_proof(
    tmp_path: Path, missing: str
) -> None:
    host = Host(tmp_path)
    host.package("omnimarket", MARKET_FILES)
    (host.work / "composed-containers").write_text("\n".join(COMPOSED) + "\n")
    host.carriers = [c for c in COMPOSED if c != missing]
    out = _identity(host, MARKET_PR="4540")
    assert f"pkg-tree omnimarket {missing} " in out.stdout, out.stdout
    assert f"pkg-tree omnimarket {missing} tree-files=2 image-files=0" in out.stdout
    probe = fakes.GOOD["probe"].split("\n", 1)[1]
    judged = pool.judge({**fakes.GOOD, "probe": out.stdout + probe})
    assert judged.checks["image_identity"] is False
    assert any(missing in n for n in judged.notes), judged.notes


def test_the_two_infra_kernels_alone_are_not_taken_for_the_omnimarket_runtime(
    tmp_path: Path,
) -> None:
    # the omnimarket#4540 receipt: both kernels healthy, no omnimarket in either
    host = Host(tmp_path)
    host.package("omnimarket", MARKET_FILES)
    (host.work / "composed-containers").write_text(f"{KERNEL_M}\n{KERNEL_E}\n")
    host.carriers = []
    out = _identity(host, MARKET_PR="4540")
    probe = fakes.GOOD["probe"].split("\n", 1)[1]
    judged = pool.judge({**fakes.GOOD, "probe": out.stdout + probe})
    assert judged.checks["image_identity"] is False


def test_an_omnimarket_proof_with_no_composed_container_list_compares_nothing(
    tmp_path: Path,
) -> None:
    # never a fallback to the two kernels: a missing list is a proof that did not
    # build the composed stack, and the judge refuses a subject with no compare
    host = Host(tmp_path)
    host.package("omnimarket", MARKET_FILES)
    host.carriers = COMPOSED
    out = _identity(host, MARKET_PR="4540")
    assert "pkg-tree " not in out.stdout
    assert "composed-containers-missing" in out.stdout
    probe = fakes.GOOD["probe"].split("\n", 1)[1]
    judged = pool.judge({**fakes.GOOD, "probe": out.stdout + probe})
    assert judged.checks["image_identity"] is False


def test_an_infra_only_proof_still_reads_the_two_kernels(tmp_path: Path) -> None:
    host = Host(tmp_path)
    host.package("omnibase_infra", {"__init__.py": "V = 1\n"})
    host.carriers = [KERNEL_M, KERNEL_E]
    out = _identity(host, INFRA_PR="5000")
    assert "identity-subject omnibase_infra" in out.stdout
    assert f"pkg-tree omnibase_infra {KERNEL_M} " in out.stdout
    assert f"pkg-tree omnibase_infra {KERNEL_E} " in out.stdout
    assert API not in out.stdout


def test_changed_path_live_reads_the_logs_of_every_container_that_runs_the_package(
    tmp_path: Path,
) -> None:
    host = Host(tmp_path)
    (host.work / "composed-containers").write_text("\n".join(COMPOSED) + "\n")
    # the node runs in the standalone writer and nowhere else
    for c in COMPOSED:
        (host.logs / c).write_text("boot\n")
    (host.logs / WRITER).write_text("consumer_group=x.omnimarket.node_a.consume\n")
    out = host.run(
        _functions("market_under_test", "proof_containers", "live_mentions")
        + "live_mentions '[.]node_a[.]'\n",
        MARKET_PR="4540",
    )
    assert out.returncode == 0, out.stderr
    for c in COMPOSED:
        assert f"  {c} mentions [.]node_a[.]: " in out.stdout, out.stdout
    assert f"  {WRITER} mentions [.]node_a[.]: 1" in out.stdout
    probe = fakes.GOOD["probe"] + out.stdout
    assert pool.judge({**fakes.GOOD, "probe": probe}).checks["changed_path_live"]


def test_changed_path_live_still_fails_when_no_container_logs_the_node(
    tmp_path: Path,
) -> None:
    host = Host(tmp_path)
    (host.work / "composed-containers").write_text("\n".join(COMPOSED) + "\n")
    for c in COMPOSED:
        (host.logs / c).write_text("Discovered contract: node_a (COMPUTE)\n")
    out = host.run(
        _functions("market_under_test", "proof_containers", "live_mentions")
        + "live_mentions '[.]node_a[.]'\n",
        MARKET_PR="4540",
    )
    probe = fakes.GOOD["probe"] + out.stdout
    judged = pool.judge({**fakes.GOOD, "probe": probe})
    assert judged.checks["changed_path_live"] is False


def test_the_composed_build_names_the_image_from_the_infra_image_and_reads_the_containers(
    tmp_path: Path,
) -> None:
    host = Host(tmp_path)
    host.package("omnimarket", MARKET_FILES)
    (host.work / "root" / "omnimarket" / "Dockerfile.runtime").write_text("FROM x\n")
    (host.work / "root" / "omnibase_infra").mkdir(exist_ok=True)
    out = host.run(
        _functions("market_under_test", "composed_build") + "composed_build\n",
        MARKET_PR="4540",
    )
    assert out.returncode == 0, out.stdout + out.stderr
    calls = host.calls.read_text(encoding="utf-8")
    build = next(ln for ln in calls.splitlines() if ln.startswith("docker build"))
    assert f"INFRA_RUNTIME_IMAGE={PROJECT}-runtime:latest" in build
    assert f"-t {PROJECT}-omnimarket-runtime:latest" in build
    assert str(host.work / "root" / "omnimarket" / "Dockerfile.runtime") in build
    read = next(ln for ln in calls.splitlines() if ln.startswith("uv run"))
    assert "image-containers" in read
    assert f"{PROJECT}-omnimarket-runtime:latest local-market" in read
    names = (host.work / "composed-containers").read_text().split()
    assert names == COMPOSED


def test_the_composed_build_refuses_a_list_that_lacks_a_kernel(tmp_path: Path) -> None:
    host = Host(tmp_path)
    host.package("omnimarket", MARKET_FILES)
    (host.work / "root" / "omnimarket" / "Dockerfile.runtime").write_text("FROM x\n")
    (host.work / "root" / "omnibase_infra").mkdir(exist_ok=True)
    host.names.write_text(f"{API}\n{WRITER}\n", encoding="utf-8")
    out = host.run(
        _functions("market_under_test", "composed_build") + "composed_build\n",
        MARKET_PR="4540",
    )
    assert out.returncode != 0
    assert "composed-containers" in out.stdout + out.stderr


def test_the_probe_and_build_phases_use_these_functions() -> None:
    text = PROVE_SH.read_text(encoding="utf-8")
    probe = text[text.index("\nprobe)") : text.index("\ntests)")]
    build = text[text.index("\nbuild)") : text.index("\nprobe)")]
    assert "identity_readback" in probe and "live_mentions" in probe
    assert "for c in $M $E; do\n      docker exec" not in probe
    assert "composed_build" in build
    assert 'generate "$BUNDLE"' in build and 'up "$BUNDLE"' in build
    assert "--composed-image" in build
