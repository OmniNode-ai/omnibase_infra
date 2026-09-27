# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import resolve_node_migration_source_ref as resolver
from scripts.ci.check_pin_reachability import Resolution, Verdict

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "resolve_node_migration_source_ref.py"
)


def _run(tmp_path: Path, body: str | None) -> subprocess.CompletedProcess[str]:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps({"pull_request": {"body": body}}), encoding="utf-8"
    )
    output_path = tmp_path / "github_output.txt"
    env = {
        **os.environ,
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_OUTPUT": str(output_path),
    }
    return subprocess.run(
        ["python3", str(SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_defaults_to_dev_without_metadata(tmp_path: Path) -> None:
    result = _run(tmp_path, "Refs OMN-15038")

    assert result.returncode == 0
    assert result.stdout.strip() == "dev"
    assert (tmp_path / "github_output.txt").read_text(encoding="utf-8") == "ref=dev\n"


def test_reads_explicit_omnimarket_source_ref_after_dev_reachability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {"pull_request": {"body": "Omnimarket-Source-Ref: jonah/landed-source-ref"}}
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    seen: list[str] = []

    def resolve_from_dev(ref: str) -> Resolution:
        seen.append(ref)
        return Resolution(Verdict.REACHABLE, "compare dev...landed = behind")

    monkeypatch.setattr(resolver, "_resolve_ref_from_dev", resolve_from_dev)

    assert resolver.main() == 0
    assert capsys.readouterr().out.strip() == "jonah/landed-source-ref"
    assert seen == ["jonah/landed-source-ref"]
    assert output_path.read_text(encoding="utf-8") == "ref=jonah/landed-source-ref\n"


def test_rejects_explicit_ref_not_reachable_from_dev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps({"pull_request": {"body": "Omnimarket-Source-Ref: jonah/unmerged"}}),
        encoding="utf-8",
    )
    output_path = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setattr(
        resolver,
        "_resolve_ref_from_dev",
        lambda ref: Resolution(
            Verdict.UNREACHABLE,
            f"compare dev...{ref} = ahead",
        ),
    )

    assert resolver.main() == 1
    assert "not reachable from omnimarket/dev" in capsys.readouterr().err
    assert not output_path.exists()


def test_unmerged_node_migration_source_ref_can_resolve_to_paired_pr_head_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_sha = "b75a8957806d918721a63a1bf72d67a1da162782"
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "pull_request": {
                    "body": "\n".join(
                        [
                            "Omnimarket-Source-Ref: jonah/omn-18079-backfill-overlay-provider",
                            "Node-Migration-Source-PR: omnimarket#2579",
                            f"Node-Migration-Source-SHA: {source_sha}",
                        ]
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setattr(
        resolver,
        "_resolve_ref_from_dev",
        lambda ref: Resolution(Verdict.UNREACHABLE, f"compare dev...{ref} = ahead"),
    )
    monkeypatch.setattr(
        resolver,
        "_api_get",
        lambda url: (
            200,
            {
                "state": "open",
                "draft": False,
                "base": {"ref": "dev"},
                "head": {
                    "ref": "jonah/omn-18079-backfill-overlay-provider",
                    "sha": source_sha,
                    "repo": {"full_name": "OmniNode-ai/omnimarket"},
                },
            },
            "HTTP 200",
        ),
    )

    assert resolver.main() == 0
    assert capsys.readouterr().out.strip() == source_sha
    assert output_path.read_text(encoding="utf-8") == f"ref={source_sha}\n"


def test_unmerged_node_migration_source_pair_does_not_require_source_ref_trailer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_sha = "b75a8957806d918721a63a1bf72d67a1da162782"
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "pull_request": {
                    "body": "\n".join(
                        [
                            "Node-Migration-Source-PR: omnimarket#2579",
                            f"Node-Migration-Source-SHA: {source_sha}",
                        ]
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setattr(
        resolver,
        "_api_get",
        lambda url: (
            200,
            {
                "state": "open",
                "draft": False,
                "base": {"ref": "dev"},
                "head": {
                    "ref": "jonah/omn-18079-backfill-overlay-provider",
                    "sha": source_sha,
                    "repo": {"full_name": "OmniNode-ai/omnimarket"},
                },
            },
            "HTTP 200",
        ),
    )

    assert resolver.main() == 0
    assert capsys.readouterr().out.strip() == source_sha
    assert output_path.read_text(encoding="utf-8") == f"ref={source_sha}\n"


def test_unmerged_node_migration_source_ref_requires_exact_paired_pr_head_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "pull_request": {
                    "body": "\n".join(
                        [
                            "Omnimarket-Source-Ref: jonah/omn-18079-backfill-overlay-provider",
                            "Node-Migration-Source-PR: #2579",
                            "Node-Migration-Source-SHA: b75a8957806d918721a63a1bf72d67a1da162782",
                        ]
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setattr(
        resolver,
        "_resolve_ref_from_dev",
        lambda ref: Resolution(Verdict.UNREACHABLE, f"compare dev...{ref} = ahead"),
    )
    monkeypatch.setattr(
        resolver,
        "_api_get",
        lambda url: (
            200,
            {
                "state": "open",
                "draft": False,
                "base": {"ref": "dev"},
                "head": {
                    "ref": "jonah/omn-18079-backfill-overlay-provider",
                    "sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "repo": {"full_name": "OmniNode-ai/omnimarket"},
                },
            },
            "HTTP 200",
        ),
    )

    assert resolver.main() == 1
    assert "head SHA does not match" in capsys.readouterr().err
    assert not output_path.exists()


def test_unmerged_node_migration_source_ref_rejects_fork_head_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_sha = "b75a8957806d918721a63a1bf72d67a1da162782"
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "pull_request": {
                    "body": "\n".join(
                        [
                            "Omnimarket-Source-Ref: jonah/omn-18079-backfill-overlay-provider",
                            "Node-Migration-Source-PR: omnimarket#2579",
                            f"Node-Migration-Source-SHA: {source_sha}",
                        ]
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setattr(
        resolver,
        "_api_get",
        lambda url: (
            200,
            {
                "state": "open",
                "draft": False,
                "base": {"ref": "dev"},
                "head": {
                    "ref": "jonah/omn-18079-backfill-overlay-provider",
                    "sha": source_sha,
                    "repo": {"full_name": "external/omnimarket"},
                },
            },
            "HTTP 200",
        ),
    )

    assert resolver.main() == 1
    assert "head repo must be OmniNode-ai/omnimarket" in capsys.readouterr().err
    assert not output_path.exists()


def test_unmerged_node_migration_source_ref_requires_both_pair_trailers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "pull_request": {
                    "body": "\n".join(
                        [
                            "Omnimarket-Source-Ref: jonah/omn-18079-backfill-overlay-provider",
                            "Node-Migration-Source-PR: #2579",
                        ]
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setattr(
        resolver,
        "_resolve_ref_from_dev",
        lambda ref: Resolution(Verdict.UNREACHABLE, f"compare dev...{ref} = ahead"),
    )

    assert resolver.main() == 1
    assert "require both" in capsys.readouterr().err
    assert not output_path.exists()


def test_rejects_explicit_ref_when_dev_reachability_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps({"pull_request": {"body": "Omnimarket-Source-Ref: jonah/unknown"}}),
        encoding="utf-8",
    )
    output_path = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))
    monkeypatch.setattr(
        resolver,
        "_resolve_ref_from_dev",
        lambda ref: Resolution(Verdict.UNDETERMINED, "HTTP 429: rate limit"),
    )

    assert resolver.main() == 1
    assert "could not prove" in capsys.readouterr().err
    assert not output_path.exists()


def test_rejects_unsafe_ref(tmp_path: Path) -> None:
    result = _run(tmp_path, "Omnimarket-Source-Ref: ../dev")

    assert result.returncode == 1
    assert "invalid omnimarket source ref" in result.stderr


# ---------------------------------------------------------------------------
# OMN-17294 defect B: the trailer was matched anywhere in the body, including
# inside fenced code blocks, and the FIRST match anywhere won.
#
# A PR body that merely *quotes* a trailer -- a runbook excerpt, a pasted CI
# log, a diff of another PR's body, a "the trailer looks like this" example --
# therefore chose the omnimarket tree the required Application Database Domain
# Enforcement job derives its TABLE grants from. Same matcher class as
# OMN-15345 (table names matched inside SQL comments).
# ---------------------------------------------------------------------------


def test_fenced_decoy_trailer_is_ignored(tmp_path: Path) -> None:
    """A trailer quoted inside a ``` fence is documentation, not a trailer."""
    result = _run(
        tmp_path,
        "Refs OMN-17294\n"
        "\n"
        "The vendoring runbook says to write:\n"
        "\n"
        "```\n"
        "Omnimarket-Source-Ref: attacker/branch\n"
        "```\n"
        "\n"
        "This PR declares no ref of its own.\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "dev"


def test_tilde_fenced_decoy_trailer_is_ignored(tmp_path: Path) -> None:
    """``~~~`` opens a code fence too (CommonMark), not only backticks."""
    result = _run(
        tmp_path,
        "Refs OMN-17294\n~~~text\nOmnimarket-Source-Ref: attacker/branch\n~~~\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "dev"


def test_fenced_decoy_does_not_outrank_the_real_trailer(tmp_path: Path) -> None:
    """First-match-anywhere let quoted text above the real trailer win."""
    result = _run(
        tmp_path,
        "Refs OMN-17294\n"
        "\n"
        "Prior art (quoted from omnibase_infra#3046):\n"
        "\n"
        "```markdown\n"
        "Omnimarket-Source-Ref: attacker/branch\n"
        "```\n"
        "\n"
        "Omnimarket-Source-Ref: dev\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "dev"


def test_indented_code_block_trailer_is_ignored(tmp_path: Path) -> None:
    """Four-space indentation is a markdown code block, and a git trailer
    lives at column 0 -- an indented line is neither a trailer nor prose."""
    result = _run(
        tmp_path,
        "Refs OMN-17294\n\n    Omnimarket-Source-Ref: attacker/branch\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "dev"


def test_inline_code_span_trailer_is_ignored(tmp_path: Path) -> None:
    """A whole-line inline code span is quoted text, not a trailer."""
    result = _run(
        tmp_path,
        "Refs OMN-17294\n`Omnimarket-Source-Ref: attacker/branch`\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "dev"


def test_conflicting_trailer_values_are_an_error(tmp_path: Path) -> None:
    """Two different declared refs is ambiguous; silent first-wins picked one."""
    result = _run(
        tmp_path,
        "Omnimarket-Source-Ref: jonah/first\nOmnimarket-Source-Ref: jonah/second\n",
    )

    assert result.returncode == 1
    assert "conflicting" in result.stderr.lower()
    assert "jonah/first" in result.stderr
    assert "jonah/second" in result.stderr


def test_conflicting_field_aliases_are_an_error(tmp_path: Path) -> None:
    """The two accepted field names must not disagree either."""
    result = _run(
        tmp_path,
        "Omnimarket-Source-Ref: jonah/first\nNode-Migration-Source-Ref: jonah/second\n",
    )

    assert result.returncode == 1
    assert "conflicting" in result.stderr.lower()


def test_repeated_identical_trailer_is_not_a_conflict(tmp_path: Path) -> None:
    """Idempotent re-stamping of the same value stays legal."""
    result = _run(
        tmp_path,
        "Omnimarket-Source-Ref: dev\nOmnimarket-Source-Ref: dev\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "dev"


def test_unterminated_fence_swallows_the_rest_of_the_body(tmp_path: Path) -> None:
    """CommonMark: an unclosed fence runs to end of document. Failing to the
    default ref is the safe direction -- the trailer is not honoured."""
    result = _run(
        tmp_path,
        "Refs OMN-17294\n```\nOmnimarket-Source-Ref: attacker/branch\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "dev"


# --- OMN-19807: merge_group events read the queued PR's live body -------------

_QUEUE_SHA = "65bdea8632" + "0" * 30
_SOURCE_SHA = "95fbd51900dcd3aad9730fbeff04bcb78fc491e1"


def _merge_group_event(tmp_path: Path, head_ref: str) -> Path:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "merge_group": {"head_ref": head_ref, "base_ref": "refs/heads/dev"},
                "repository": {"full_name": "OmniNode-ai/omnibase_infra"},
            }
        ),
        encoding="utf-8",
    )
    return event_path


def _paired_api(queued_body: str | None, queued_status: int = 200):  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def api_get(url: str) -> tuple[int, dict[str, object] | None, str]:
        calls.append(url)
        if url.endswith("/repos/OmniNode-ai/omnibase_infra/pulls/4165"):
            if queued_status != 200:
                return queued_status, None, f"HTTP {queued_status}"
            return 200, {"body": queued_body}, "HTTP 200"
        if url.endswith("/repos/OmniNode-ai/omnimarket/pulls/2953"):
            return (
                200,
                {
                    "state": "open",
                    "draft": False,
                    "base": {"ref": "dev"},
                    "head": {
                        "ref": "jonah/omn-19716-topic-activity-projection",
                        "sha": _SOURCE_SHA,
                        "repo": {"full_name": "OmniNode-ai/omnimarket"},
                    },
                },
                "HTTP 200",
            )
        raise AssertionError(f"unexpected API read {url}")

    return api_get, calls


@pytest.mark.parametrize(
    "head_ref",
    [
        f"refs/heads/gh-readonly-queue/dev/pr-4165-{_QUEUE_SHA}",
        f"gh-readonly-queue/dev/pr-4165-{_QUEUE_SHA}",
    ],
)
def test_merge_group_reads_the_queued_pr_body_and_its_paired_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    head_ref: str,
) -> None:
    body = "\n".join(
        [
            "Node-Migration-Source-PR: omnimarket#2953",
            f"Node-Migration-Source-SHA: {_SOURCE_SHA}",
        ]
    )
    api_get, calls = _paired_api(body)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(_merge_group_event(tmp_path, head_ref)))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out.txt"))
    monkeypatch.setattr(resolver, "_api_get", api_get)

    assert resolver.main() == 0
    assert capsys.readouterr().out.strip() == _SOURCE_SHA
    assert calls[0].endswith("/repos/OmniNode-ai/omnibase_infra/pulls/4165")


def test_merge_group_without_trailers_still_resolves_dev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api_get, _ = _paired_api("Refs OMN-19807")
    monkeypatch.setenv(
        "GITHUB_EVENT_PATH",
        str(
            _merge_group_event(
                tmp_path, f"refs/heads/gh-readonly-queue/dev/pr-4165-{_QUEUE_SHA}"
            )
        ),
    )
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out.txt"))
    monkeypatch.setattr(resolver, "_api_get", api_get)

    assert resolver.main() == 0
    assert capsys.readouterr().out.strip() == "dev"


def test_merge_group_with_unreadable_pr_body_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api_get, _ = _paired_api(None, queued_status=404)
    monkeypatch.setenv(
        "GITHUB_EVENT_PATH",
        str(
            _merge_group_event(
                tmp_path, f"refs/heads/gh-readonly-queue/dev/pr-4165-{_QUEUE_SHA}"
            )
        ),
    )
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out.txt"))
    monkeypatch.setattr(resolver, "_api_get", api_get)

    assert resolver.main() == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == ""
    assert "fail-closed" in captured.err


def test_merge_group_with_a_foreign_head_ref_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api_get, calls = _paired_api("Refs OMN-19807")
    monkeypatch.setenv(
        "GITHUB_EVENT_PATH", str(_merge_group_event(tmp_path, "refs/heads/feature/x"))
    )
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out.txt"))
    monkeypatch.setattr(resolver, "_api_get", api_get)

    assert resolver.main() == 1
    assert "not a merge-queue ref" in capsys.readouterr().err
    assert calls == []
