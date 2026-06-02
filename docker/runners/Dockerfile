# GitHub Actions self-hosted runner image for OmniNode CI
# Ticket: OMN-3275 / Epic: OMN-3273
# Versioned image contract + prebuilt env + bound identity: OMN-12567
#
# NOT pre-baked: ruff, mypy — repos pin via pyproject.toml; workflows call `uv run ruff`.
# Pre-baking the *tools* causes tool drift. The repo dependency *environment*,
# by contrast, is pinned by uv.lock and baked as a prebuilt shared CI env so the
# happy path resolves zero `uv sync` (OMN-12564 canary, OMN-12567 image
# contract). Build args MUST match docker/runners/runner-image.lock.json — CI
# verifies the bound identity via scripts/ci/runner_image_identity.py.

ARG RUNNER_VERSION=2.323.0
ARG GH_VERSION=2.67.0
ARG KUBECTL_VERSION=1.32.1
# uv pinned to the shared CI env (canary) version so the image binding is truthful.
ARG UV_VERSION=0.6.14
# Bound runner image identity (binding, not a human label). Pass via:
#   --build-arg OMNI_RUNNER_IMAGE_IDENTITY=$(jq -r .identity_digest docker/runners/runner-image.lock.json)
#   --build-arg OMNI_RUNNER_IMAGE_VERSION=$(jq -r .image_version docker/runners/runner-image.lock.json)
ARG OMNI_RUNNER_IMAGE_IDENTITY=unbound
ARG OMNI_RUNNER_IMAGE_VERSION=0
# Prebuilt shared CI env root baked into the image (matches ensure_ci_env.sh).
ARG OMNI_CI_ENV_ROOT=/home/runner/.cache/omni/ci-envs

# Pinned digest for reproducible builds — update via: docker pull ubuntu:22.04
FROM ubuntu:22.04@sha256:3ba65aa20f86a0fad9df2b2c259c613df006b2e6d0bfcc8a146afb8c525a9751

ARG RUNNER_VERSION
ARG GH_VERSION
ARG KUBECTL_VERSION
ARG UV_VERSION
ARG OMNI_RUNNER_IMAGE_IDENTITY
ARG OMNI_RUNNER_IMAGE_VERSION
ARG OMNI_CI_ENV_ROOT

# Version labels — read by runner-status skill. These remain human-facing
# labels; the *authoritative* version is the bound identity below, which CI
# verifies against docker/runners/runner-image.lock.json.
LABEL org.omninode.runner.version="${RUNNER_VERSION}"
LABEL org.omninode.gh.version="${GH_VERSION}"
LABEL org.omninode.kubectl.version="${KUBECTL_VERSION}"
LABEL org.omninode.uv.version="${UV_VERSION}"
# Bound identity (binding, not a label-by-convention): the digest folds base
# image + manifest + python + uv + shared-env + image version (OMN-12567).
LABEL org.omninode.runner.image.identity="${OMNI_RUNNER_IMAGE_IDENTITY}"
LABEL org.omninode.runner.image.version="${OMNI_RUNNER_IMAGE_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive
ENV RUNNER_HOME=/home/runner/actions-runner
# Persist the bound identity into the runtime env so every job can emit it as
# startup evidence (scripts/ci/runner_image_identity.py --mode emit reads the
# committed lock; these vars let workloads cross-check the running image).
ENV OMNI_RUNNER_IMAGE_IDENTITY=${OMNI_RUNNER_IMAGE_IDENTITY}
ENV OMNI_RUNNER_IMAGE_VERSION=${OMNI_RUNNER_IMAGE_VERSION}
ENV OMNI_CI_ENV_ROOT=${OMNI_CI_ENV_ROOT}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    jq \
    tar \
    unzip \
    libicu-dev \
    libkrb5-3 \
    zlib1g \
    libssl3 \
    lsb-release \
    gnupg \
    apt-transport-https \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 via deadsnakes PPA
RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python

# Install uv
RUN curl -LsSf "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    | tar -xz -C /usr/local/bin --strip-components=1 uv-x86_64-unknown-linux-gnu/uv \
    && chmod +x /usr/local/bin/uv

# Install Docker CLI (no daemon — socket mounted at runtime)
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Install gh CLI
RUN curl -fsSL "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz" \
    | tar -xz -C /usr/local/bin --strip-components=2 "gh_${GH_VERSION}_linux_amd64/bin/gh" \
    && chmod +x /usr/local/bin/gh

# Install kubectl
RUN curl -fsSLo /usr/local/bin/kubectl \
    "https://dl.k8s.io/release/v${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && chmod +x /usr/local/bin/kubectl

# Create non-root runner user and add to docker group for socket access
RUN groupadd --gid 1001 runner \
    && useradd --uid 1001 --gid runner --shell /bin/bash --create-home runner \
    && groupadd --force docker \
    && usermod -aG docker runner

# Download and SHA256-verify GitHub Actions runner binary
# SHA256 for actions-runner-linux-x64-2.323.0.tar.gz
ENV RUNNER_SHA256=0dbc9bf5a58620fc52cb6cc0448abcca964a8d74b5f39773b7afcad9ab691e19

RUN mkdir -p "${RUNNER_HOME}" \
    && cd "${RUNNER_HOME}" \
    && curl -fsSLo runner.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    && echo "${RUNNER_SHA256}  runner.tar.gz" | sha256sum --check --strict \
    && tar xzf runner.tar.gz \
    && rm runner.tar.gz \
    && ./bin/installdependencies.sh \
    && chown -R runner:runner "${RUNNER_HOME}"

# Credential cache directory (for entrypoint.sh re-registration cache)
RUN mkdir -p /home/runner/.runner-creds && chown runner:runner /home/runner/.runner-creds

# --- Bound identity contract + prebuilt shared CI env (OMN-12567) ----------
# The bound identity lock is the authority for "runner image vN". It is copied
# into the image at a stable path so runner-side tooling and CI startup
# evidence can read it without a checkout.
COPY runner-image.lock.json /etc/omni/runner-image.lock.json
RUN chmod 0444 /etc/omni/runner-image.lock.json
#
# Prebuilt env build context (root-relative): the dependency manifest + the
# canary env scripts are copied so the image can bake the repo-specific .venv
# as part of the image contract. The shared env is materialised by the same
# ensure_ci_env.sh the canary uses (OMN-12564), so a baked env and a runtime
# env are byte-identical at a given lock digest. Built under the runner user so
# the env directory ownership matches the runtime user.
COPY prebuilt-env-context/pyproject.toml /opt/omni/prebuilt/pyproject.toml
COPY prebuilt-env-context/uv.lock /opt/omni/prebuilt/uv.lock
COPY prebuilt-env-context/ci/ /opt/omni/prebuilt/scripts/ci/
# OMN-12584: chown the *whole* runner cache dir, not only the ci-envs subtree.
# `mkdir -p "${OMNI_CI_ENV_ROOT}/omnibase_infra"` runs as root and creates the
# parent /home/runner/.cache owned by root:root. The bake step below runs the
# shared-env build as USER runner, and uv writes its cache to its default
# ~/.cache/uv = /home/runner/.cache/uv. If only the ci-envs subtree is
# runner-owned, that uv-cache mkdir fails with EACCES and the prebuilt-env bake
# (the core of the OMN-12567 image contract) never completes. Widening the chown
# to /home/runner/.cache fixes the bake. This is identity-neutral: the Dockerfile
# is not part of the runner_image_identity binding, so runner-image.lock.json's
# identity_digest is unchanged.
RUN mkdir -p "${OMNI_CI_ENV_ROOT}/omnibase_infra" /home/runner/.cache \
    && chown -R runner:runner /opt/omni/prebuilt /home/runner/.cache
USER runner
RUN cd /opt/omni/prebuilt \
    && OMNI_CI_ENV_ROOT="${OMNI_CI_ENV_ROOT}" \
       OMNI_CI_REPO=omnibase_infra \
       PYTHON_VERSION="3.12" \
       UV_VERSION="${UV_VERSION}" \
       GITHUB_ENV=/dev/null \
       bash scripts/ci/ensure_ci_env.sh \
    && echo "prebuilt shared CI env baked under ${OMNI_CI_ENV_ROOT}/omnibase_infra"
USER root

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY runner-job-started.sh /usr/local/bin/runner-job-started.sh
COPY healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/runner-job-started.sh /usr/local/bin/healthcheck.sh

# Install gosu for privilege de-escalation in entrypoint.
# The entrypoint starts as root to fix Docker socket GID, then drops to runner.
RUN curl -fsSLo /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/1.17/gosu-amd64" \
    && chmod +x /usr/local/bin/gosu \
    && gosu nobody true

# Run as root — entrypoint fixes Docker socket GID then drops to runner via gosu
WORKDIR ${RUNNER_HOME}

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
