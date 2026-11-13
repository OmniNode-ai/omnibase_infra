# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fill the self-hosted runner action archive cache (OMN-24299).

The runner (v2.338.2, read from its Runner.Common.dll) reads
``ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE`` and, when a job step uses
``owner/repo@<resolved sha>``, looks for
``<cache>/<owner>_<repo>/<sha>.tar.gz`` and copies it instead of downloading from
GitHub. This handler writes those archives: the action references come from the
``uses:`` lines of the canonical clones' workflows, the commit comes from the
pinned sha or from the clone / a once-per-run bare source mirror, and the archive
is ``git archive`` output with one top-level directory like GitHub's own tarball.

Every git call is an argv list with the configured timeout, never a shell. A ref
that cannot be read is a typed failure naming the ref; the other refs are still
filled.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_action_archive_cache_config import (
    ModelActionArchiveCacheConfig,
)
from omnibase_infra.models.model_action_archive_cache_failure import (
    ModelActionArchiveCacheFailure,
)
from omnibase_infra.models.model_action_archive_cache_fill_request import (
    ModelActionArchiveCacheFillRequest,
)
from omnibase_infra.models.model_action_archive_cache_fill_result import (
    ModelActionArchiveCacheFillResult,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

_USES_RE = re.compile(r"""uses:\s*['"]?([^'"\s#]+)""")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_REF_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
_WORKFLOW_DIRS = (".github/workflows/", ".github/actions/")
_REUSABLE_WORKFLOW_MARKER = "/.github/workflows/"


def _failure(ref: str, code: str, detail: str) -> ModelActionArchiveCacheFailure:
    return ModelActionArchiveCacheFailure(ref=ref, reason_code=code, detail=detail)


class HandlerActionArchiveCacheFill:
    """Read workflow sources, resolve refs, write one archive per resolved sha."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelActionArchiveCacheFillRequest
    ) -> ModelActionArchiveCacheFillResult:
        try:
            fleet = load_runner_fleet_config(Path(request.config_path))
        except FileNotFoundError:
            return self._refusal(f"config not found: {request.config_path}")
        cfg = fleet.action_archive_cache
        if cfg is None:
            return self._refusal("action_archive_cache block absent")
        if not cfg.active:
            return self._refusal("action_archive_cache is not active")

        if fleet.git_mirror is None:
            return self._refusal(
                "git_mirror block absent: it names the repositories scanned"
            )
        staging = Path(cfg.staging_root).expanduser()
        clone_root = Path(request.clone_root)
        failures: list[ModelActionArchiveCacheFailure] = []
        wanted = self._collect_refs(cfg, fleet.git_mirror.repos, clone_root, failures)

        filled: list[str] = []
        present: list[str] = []
        mirror_errors: dict[tuple[str, str], str | None] = {}
        for (owner, repo, ref), written in wanted.items():
            key = f"{owner}/{repo}@{ref}"
            target_dir = staging / f"{owner}_{repo}"
            if _HEX40_RE.match(ref) and self._is_filled(target_dir / f"{ref}.tar.gz"):
                present.append(key)
                continue
            if owner == cfg.local_owner:
                source = clone_root / repo
                mirror_error = None
            else:
                source = staging / ".sources" / f"{owner}_{repo}.git"
                if (owner, repo) not in mirror_errors:
                    mirror_errors[(owner, repo)] = self._refresh_mirror(
                        cfg, source, f"{owner}/{repo}"
                    )
                mirror_error = mirror_errors[(owner, repo)]
            sha = self._resolve_sha(source, ref, cfg.fetch_timeout_seconds)
            if sha is None:
                failures.append(
                    _failure(written, "SOURCE_UNREACHABLE", mirror_error)
                    if mirror_error
                    else _failure(
                        written,
                        "REF_UNRESOLVED",
                        f"no commit for {ref} in {owner}/{repo}",
                    )
                )
                continue
            target = target_dir / f"{sha}.tar.gz"
            if self._is_filled(target):
                present.append(f"{owner}/{repo}@{sha}")
            elif request.dry_run:
                filled.append(f"{owner}/{repo}@{sha}")
            else:
                error = self._write_archive(
                    source, owner, repo, sha, target, cfg.fetch_timeout_seconds
                )
                if error:
                    failures.append(_failure(written, "ARCHIVE_FAILED", error))
                else:
                    filled.append(f"{owner}/{repo}@{sha}")

        return ModelActionArchiveCacheFillResult(
            success=not failures,
            staging_root=str(staging),
            filled=tuple(filled),
            present=tuple(present),
            failed=tuple(failures),
            error_message="; ".join(f"{f.reason_code} {f.ref}" for f in failures),
        )

    @staticmethod
    def _refusal(detail: str) -> ModelActionArchiveCacheFillResult:
        return ModelActionArchiveCacheFillResult(
            success=False,
            staging_root="",
            failed=(_failure("", "CONFIG_ABSENT", detail),),
            error_message=detail,
        )

    @staticmethod
    def _is_filled(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    @staticmethod
    def _git(
        repo_dir: Path, args: list[str], timeout: int
    ) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                ["git", "-C", str(repo_dir), *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None

    def _collect_refs(
        self,
        cfg: ModelActionArchiveCacheConfig,
        sources: tuple[str, ...],
        clone_root: Path,
        failures: list[ModelActionArchiveCacheFailure],
    ) -> dict[tuple[str, str, str], str]:
        """Every distinct remote (owner, repo, ref) the sources use -> as written."""
        timeout = cfg.fetch_timeout_seconds
        wanted: dict[tuple[str, str, str], str] = {}
        for name in sources:
            clone = clone_root / name
            head = self._git(
                clone,
                ["symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
                timeout,
            )
            ref = (
                head.stdout.strip()
                if head and head.returncode == 0 and head.stdout.strip()
                else "HEAD"
            )
            listing = self._git(clone, ["ls-tree", "-r", "--name-only", ref], timeout)
            if listing is None or listing.returncode != 0:
                failures.append(
                    _failure(
                        name,
                        "WORKFLOW_SOURCE_MISSING",
                        f"{name} is not a readable clone",
                    )
                )
                continue
            for path in listing.stdout.splitlines():
                if not path.endswith((".yml", ".yaml")) or not path.startswith(
                    _WORKFLOW_DIRS
                ):
                    continue
                shown = self._git(clone, ["show", f"{ref}:{path}"], timeout)
                if shown is None or shown.returncode != 0:
                    continue
                for match in _USES_RE.finditer(shown.stdout):
                    parsed = self._parse_uses(match.group(1))
                    if parsed is not None:
                        wanted.setdefault(parsed, match.group(1))
        return wanted

    @staticmethod
    def _parse_uses(value: str) -> tuple[str, str, str] | None:
        if value.startswith(("./", "docker://")) or "@" not in value:
            return None
        path, ref = value.rsplit("@", 1)
        if _REUSABLE_WORKFLOW_MARKER in f"/{path}":
            return None
        segments = path.split("/")
        if len(segments) < 2 or not segments[0] or not segments[1]:
            return None
        if not _REF_RE.match(ref):
            return None
        return segments[0], segments[1], ref

    def _refresh_mirror(
        self, cfg: ModelActionArchiveCacheConfig, mirror: Path, repo: str
    ) -> str | None:
        """Fetch the source mirror once; the error text, or None when it is current."""
        timeout = cfg.fetch_timeout_seconds
        if not mirror.exists():
            mirror.parent.mkdir(parents=True, exist_ok=True)
            made = self._git(
                mirror.parent, ["init", "--bare", "--quiet", str(mirror)], timeout
            )
            if made is None or made.returncode != 0:
                return f"cannot create source mirror for {repo}"
        done = self._git(
            mirror,
            [
                "fetch",
                "--force",
                "--quiet",
                cfg.remote_url_template.replace("{repo}", repo),
                "+refs/heads/*:refs/heads/*",
                "+refs/tags/*:refs/tags/*",
            ],
            timeout,
        )
        if done is None:
            return f"fetch of {repo} timed out after {timeout}s"
        if done.returncode != 0:
            return f"fetch of {repo} failed: {done.stderr.strip()[:200]}"
        return None

    def _resolve_sha(self, source: Path, ref: str, timeout: int) -> str | None:
        if _HEX40_RE.match(ref):
            done = self._git(source, ["cat-file", "-e", f"{ref}^{{commit}}"], timeout)
            return ref if done is not None and done.returncode == 0 else None
        for candidate in (
            f"refs/tags/{ref}^{{commit}}",
            f"refs/remotes/origin/{ref}^{{commit}}",
            f"refs/heads/{ref}^{{commit}}",
        ):
            done = self._git(source, ["rev-parse", "--verify", candidate], timeout)
            if done is not None and done.returncode == 0 and done.stdout.strip():
                return done.stdout.strip()
        return None

    def _write_archive(
        self,
        source: Path,
        owner: str,
        repo: str,
        sha: str,
        target: Path,
        timeout: int,
    ) -> str | None:
        """Write the archive atomically; the error text, or None on success."""
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.partial")
        try:
            with partial.open("wb") as out:
                done = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(source),
                        "archive",
                        "--format=tar.gz",
                        f"--prefix={owner}-{repo}-{sha[:7]}/",
                        sha,
                    ],
                    stdout=out,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=timeout,
                )
            if done.returncode != 0:
                return f"git archive exited {done.returncode}: {done.stderr.decode(errors='replace').strip()[:200]}"
            if partial.stat().st_size == 0:
                return "git archive wrote an empty file"
            partial.replace(target)
        except (subprocess.TimeoutExpired, OSError) as exc:
            return f"git archive failed: {exc}"
        finally:
            partial.unlink(missing_ok=True)
        return None
