# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the contract's published-source proof against GitHub response failures.

The pinned verifier's contents shim omits sha. Its valid response must still
permit the source proof; a blob existing elsewhere must not prove path membership.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[2]
REPO = "OmniNode-ai/omnibase_infra"
SOURCE_PATHS = (
    "src/omnibase_infra/handlers/handler_developer_principal_plan.py",
    "deploy/local/developer-principal-grants.yaml",
    "deploy/local/machine-principal-bindings.yaml",
    "deploy/local/machine-principal-derived-grants.json",
)

FAKE_GH = r"""import json, os, subprocess, sys
from pathlib import Path
responses = json.loads(Path(os.environ["SOURCE_PROOF_RESPONSES"]).read_text())
args = sys.argv[1:]
if not args or args[0] != "api" or len(args) < 2:
    sys.exit(2)
response = responses.get(args[1])
if response is None or response.get("exit_code", 0):
    print("fixture GitHub read refused", file=sys.stderr)
    sys.exit(1)
document = json.dumps(response["document"])
if "--jq" in args:
    expression = args[args.index("--jq") + 1]
    result = subprocess.run([os.environ["SOURCE_PROOF_JQ"], "-r", expression],
                            input=document, text=True, capture_output=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)
print(document)
"""


def _git(value: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", value],
        cwd=REPO_ROOT,
        env=scrub_git_location_env(os.environ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "missing_commit",
        "wrong_commit",
        "wrong_root",
        "missing_path",
        "wrong_blob",
        "non_blob",
        "truncated",
        "missing_truncated",
        "api_error",
    ],
)
def test_published_source_proves_commit_and_exact_paths(
    tmp_path: Path, mode: str
) -> None:
    jq = shutil.which("jq")
    assert jq is not None, "fixture requires jq to execute gh's actual --jq expressions"
    ref, root = _git("HEAD"), _git("HEAD^{tree}")
    entries = [
        {"path": path, "type": "blob", "sha": _git(f"{ref}:{path}")}
        for path in SOURCE_PATHS
    ]
    commit = {"sha": ref, "tree": {"sha": root}}
    tree = {"sha": root, "truncated": False, "tree": entries}
    prefix = f"repos/{REPO}"
    commit_endpoint = f"{prefix}/git/commits/{ref}"
    responses = {
        commit_endpoint: {"document": commit},
        f"{prefix}/git/trees/{root}?recursive=1": {"document": tree},
    }
    for entry in entries:
        path, blob = entry["path"], entry["sha"]
        # This is the pinned shim's real shape: content exists, sha is absent.
        responses[f"{prefix}/contents/{path}?ref={ref}"] = {
            "document": {
                "type": "file",
                "encoding": "base64",
                "path": path,
                "size": 0,
                "content": "",
            }
        }
        responses[f"{prefix}/git/blobs/{blob}"] = {"document": {"sha": blob}}
    if mode in {"missing_commit", "api_error"}:
        responses[commit_endpoint] = {"exit_code": 1}
    elif mode == "wrong_commit":
        commit["sha"] = "0" * 40
    elif mode == "wrong_root":
        commit["tree"]["sha"] = "0" * 40
    elif mode == "missing_path":
        entries.pop()
    elif mode == "wrong_blob":
        entries[-1]["sha"] = "0" * 40
    elif mode == "non_blob":
        entries[-1]["type"] = "tree"
    elif mode == "truncated":
        tree["truncated"] = True
    elif mode == "missing_truncated":
        tree.pop("truncated")
    response_path = tmp_path / "responses.json"
    response_path.write_text(json.dumps(responses))
    binary = tmp_path / "gh"
    binary.write_text(f"#!{sys.executable}\n" + FAKE_GH)
    binary.chmod(0o755)
    env = scrub_git_location_env(os.environ)
    # User shell startup files must not replace this fixture provider environment.
    env.pop("BASH_ENV", None)
    env.pop("ENV", None)
    env.pop("SSH_CLIENT", None)
    env.pop("SSH_CONNECTION", None)
    env.update(
        SOURCE_PROOF_RESPONSES=str(response_path),
        SOURCE_PROOF_JQ=jq,
        PATH=str(tmp_path) + os.pathsep + env.get("PATH", ""),
    )
    contract = yaml.safe_load((REPO_ROOT / "contracts/OMN-20825.yaml").read_text())
    evidence = next(
        item
        for item in contract["dod_evidence"]
        if item["id"] == "dod-omn20825-deploy-source"
    )
    command = evidence["checks"][0]["check_value"]
    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", command],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    diagnostic = f"{mode}: rc={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}"
    if mode == "valid":
        assert result.returncode == 0, diagnostic
    else:
        assert result.returncode != 0, diagnostic
