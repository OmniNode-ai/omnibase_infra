#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
set -euo pipefail

repo_root="$(pwd)"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

env_root="${OMNI_CI_ENV_ROOT:-/home/runner/.cache/omni/ci-envs}"
repo_name="${OMNI_CI_REPO:-omnibase_infra}"
python_version="${PYTHON_VERSION:-3.12}"
uv_version="${UV_VERSION:-0.6.14}"
install_args="${OMNI_CI_ENV_INSTALL_ARGS:---frozen --all-extras --all-groups --no-install-project}"
digest_extra="${OMNI_CI_ENV_DIGEST_EXTRA:-}"
sync_attempts="${OMNI_CI_ENV_SYNC_ATTEMPTS:-5}"
retry_delay_seconds="${OMNI_CI_ENV_SYNC_RETRY_DELAY_SECONDS:-10}"

if [[ -n "${pythonLocation:-}" && -x "${pythonLocation}/bin/python" ]]; then
  python_bin="${pythonLocation}/bin/python"
else
  python_bin="$(command -v python3)"
fi

digest="$(
  "${python_bin}" "${script_dir}/ci_env_digest.py" \
    --repo-root "${repo_root}" \
    --python-version "${python_version}" \
    --uv-version "${uv_version}" \
    --install-args "${install_args}" \
    --extra "${digest_extra}"
)"

repo_env_root="${env_root}/${repo_name}"
env_dir="${repo_env_root}/${digest}"
venv_dir="${env_dir}/.venv"
workspace_venv="${repo_root}/.venv"
wrapper_parent="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
wrapper_dir="${wrapper_parent%/}/omni-ci-bin-${digest}"
metadata_root="${wrapper_parent%/}/omni-ci-metadata-${digest}"
real_uv="$(command -v uv)"
manifest_path="${env_dir}/manifest.json"
lock_dir="${repo_env_root}/.locks"

ready() {
  [[ -x "${venv_dir}/bin/python" && -f "${manifest_path}" ]]
}

publish_env() {
  {
    echo "OMNI_CI_ENV_DIGEST=${digest}"
    echo "OMNI_CI_ENV_DIR=${env_dir}"
    echo "VIRTUAL_ENV=${workspace_venv}"
    echo "UV_PROJECT_ENVIRONMENT=${workspace_venv}"
    echo "UV_NO_SYNC=1"
    echo "OMNI_CI_SHARED_UV_RUN_DIRECT=1"
    echo "PYTHONDONTWRITEBYTECODE=1"
    echo "PATH=${wrapper_dir}:${workspace_venv}/bin:${PATH}"
    if [[ -n "${PYTHONPATH:-}" ]]; then
      echo "PYTHONPATH=${metadata_root}:${repo_root}/src:${PYTHONPATH}"
    else
      echo "PYTHONPATH=${metadata_root}:${repo_root}/src"
    fi
  } >> "${GITHUB_ENV}"
  echo "Shared CI env: ${env_dir}"
}

link_workspace_venv() {
  if [[ -e "${workspace_venv}" || -L "${workspace_venv}" ]]; then
    if [[ ! -L "${workspace_venv}" ]]; then
      echo "::error::workspace .venv exists and is not a symlink: ${workspace_venv}"
      exit 2
    fi
    ln -sfn "${venv_dir}" "${workspace_venv}"
  else
    ln -s "${venv_dir}" "${workspace_venv}"
  fi
}

write_uv_wrapper() {
  mkdir -p "${wrapper_dir}"
  cat > "${wrapper_dir}/uv" <<EOF
#!/usr/bin/env bash
set -euo pipefail

real_uv="${real_uv}"
workspace_venv="${workspace_venv}"

if [[ "\${OMNI_CI_SHARED_UV_RUN_DIRECT:-0}" == "1" && "\${1:-}" == "run" ]]; then
  shift
  if [[ "\${1:-}" == "--" ]]; then
    shift
  fi

  if [[ "\$#" -gt 0 && "\${1:0:1}" != "-" ]]; then
    cmd="\${1}"
    shift
    if [[ -x "\${workspace_venv}/bin/\${cmd}" ]]; then
      exec "\${workspace_venv}/bin/\${cmd}" "\$@"
    fi
    exec "\${cmd}" "\$@"
  fi
fi

exec "\${real_uv}" "\$@"
EOF
  chmod +x "${wrapper_dir}/uv"
}

write_project_metadata() {
  rm -rf "${metadata_root}"
  mkdir -p "${metadata_root}"
  "${python_bin}" - "${repo_root}" "${metadata_root}" <<'PY'
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

repo_root = Path(sys.argv[1])
metadata_root = Path(sys.argv[2])
project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))[
    "project"
]
name = project["name"]
version = project["version"]
dist_name = re.sub(r"[-_.]+", "_", name)
dist_info = metadata_root / f"{dist_name}-{version}.dist-info"
dist_info.mkdir(parents=True, exist_ok=True)
entry_point_groups: dict[str, dict[str, str]] = {}
scripts = project.get("scripts", {})
if scripts:
    entry_point_groups["console_scripts"] = scripts
entry_point_groups.update(project.get("entry-points", {}))

(dist_info / "METADATA").write_text(
    f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n",
    encoding="utf-8",
)
(dist_info / "entry_points.txt").write_text(
    "\n".join(
        line
        for group, entries in entry_point_groups.items()
        for line in (
            f"[{group}]",
            *(f"{entry_name} = {target}" for entry_name, target in entries.items()),
            "",
        )
    ),
    encoding="utf-8",
)
(dist_info / "RECORD").write_text("", encoding="utf-8")
PY
}

mkdir -p "${lock_dir}"

if ! ready; then
  if command -v flock >/dev/null 2>&1; then
    lock_path="${lock_dir}/${digest}.lock"
    exec 9>"${lock_path}"
    flock 9
  else
    lock_path="${lock_dir}/${digest}.lockdir"
    until mkdir "${lock_path}" 2>/dev/null; do
      echo "Waiting for shared CI env lock: ${lock_path}"
      sleep 2
    done
    trap 'rmdir "${lock_path}" 2>/dev/null || true' EXIT
  fi

  if ! ready; then
    rm -rf "${env_dir}"
    mkdir -p "${env_dir}"

    export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
    export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"
    export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"
    export UV_PROJECT_ENVIRONMENT="${venv_dir}"
    export UV_NO_CACHE="${UV_NO_CACHE:-0}"

    git config --global http.version HTTP/1.1
    if [[ -n "${GIT_FETCH_TOKEN:-}" ]]; then
      export GIT_CONFIG_COUNT=1
      export GIT_CONFIG_KEY_0="url.https://x-access-token:${GIT_FETCH_TOKEN}@github.com/.insteadOf"
      export GIT_CONFIG_VALUE_0="https://github.com/"
      echo "git github.com fetches: authenticated (x-access-token)"
    fi

    read -r -a install_argv <<< "${install_args}"
    if ! [[ "${sync_attempts}" =~ ^[1-9][0-9]*$ ]]; then
      echo "::error::OMNI_CI_ENV_SYNC_ATTEMPTS must be a positive integer (got: ${sync_attempts})"
      exit 2
    fi
    if ! [[ "${retry_delay_seconds}" =~ ^[0-9]+$ ]]; then
      echo "::error::OMNI_CI_ENV_SYNC_RETRY_DELAY_SECONDS must be a non-negative integer (got: ${retry_delay_seconds})"
      exit 2
    fi

    echo "Building shared CI env ${env_dir}"
    echo "uv version: $(uv --version)"
    echo "install args: ${install_args}"
    echo "uv sync attempts: ${sync_attempts}"
    echo "uv sync retry delay seconds: ${retry_delay_seconds}"
    attempt=1
    until uv sync "${install_argv[@]}"; do
      status=$?
      if [[ "${attempt}" -ge "${sync_attempts}" ]]; then
        echo "::error::shared CI env uv sync failed after ${attempt} attempt(s)"
        exit "${status}"
      fi

      sleep_seconds=$((retry_delay_seconds * attempt))
      echo "::warning::shared CI env uv sync attempt ${attempt}/${sync_attempts} failed with exit ${status}; retrying in ${sleep_seconds}s"
      sleep "${sleep_seconds}"
      attempt=$((attempt + 1))
    done

    cat > "${manifest_path}" <<EOF
{
  "schema": 1,
  "repo": "${repo_name}",
  "digest": "${digest}",
  "python_version": "${python_version}",
  "uv_version": "${uv_version}",
  "install_args": "${install_args}"
}
EOF
    chmod -R a-w "${env_dir}"
  fi
fi

link_workspace_venv
write_uv_wrapper
write_project_metadata
publish_env
