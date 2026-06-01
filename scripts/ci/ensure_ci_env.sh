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
wrapper_dir="${repo_root}/.omni-ci-bin"
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
      echo "PYTHONPATH=${repo_root}/src:${PYTHONPATH}"
    else
      echo "PYTHONPATH=${repo_root}/src"
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
    echo "Building shared CI env ${env_dir}"
    echo "uv version: $(uv --version)"
    echo "install args: ${install_args}"
    uv sync "${install_argv[@]}"

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
publish_env
