#!/usr/bin/env bash
#
# Install Git Hooks - File Change Event Publisher
#
# Installs pre-push git hook that publishes file change events to Kafka.
# Supports installation in multiple repositories with environment configuration.
#
# Usage:
#   # Install in current repository
#   ./scripts/install_git_hooks.sh
#
#   # Install in specific repository
#   ./scripts/install_git_hooks.sh /path/to/repo
#
#   # Install in multiple repositories
#   ./scripts/install_git_hooks.sh /path/to/repo1 /path/to/repo2
#
#   # Uninstall hooks
#   ./scripts/install_git_hooks.sh --uninstall
#
# Environment Variables:
#   KAFKA_BOOTSTRAP_SERVERS: Kafka broker address (default: localhost:29092)
#   KAFKA_ENABLE_LOGGING: Enable Kafka event logging (default: true)
#   GIT_HOOK_TIMEOUT: Max execution time in seconds (default: 2)
#   GIT_HOOK_DEBUG: Enable debug logging (default: false)
#
# Exit Codes:
#   0: Success
#   1: Error (invalid repository, installation failed)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_HOOKS_DIR="${SCRIPT_DIR}/git_hooks"
HOOK_SCRIPT="${GIT_HOOKS_DIR}/pre_push_event_publisher.py"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check if running in git repository
is_git_repo() {
    local repo_path="${1:-.}"
    if [ -d "${repo_path}/.git" ]; then
        return 0
    else
        return 1
    fi
}

# Get repository name from path
get_repo_name() {
    local repo_path="${1:-.}"
    basename "$(cd "${repo_path}" && git rev-parse --show-toplevel 2>/dev/null || echo "${repo_path}")"
}

# Install hook in repository
install_hook() {
    local repo_path="${1:-.}"
    local repo_name

    # Validate repository
    if ! is_git_repo "${repo_path}"; then
        log_error "Not a git repository: ${repo_path}"
        return 1
    fi

    repo_name=$(get_repo_name "${repo_path}")
    log_info "Installing git hook in ${repo_name} (${repo_path})"

    # Get git hooks directory
    local hooks_dir="${repo_path}/.git/hooks"
    if [ ! -d "${hooks_dir}" ]; then
        log_error "Git hooks directory not found: ${hooks_dir}"
        return 1
    fi

    # Check if hook already exists
    local hook_path="${hooks_dir}/pre-push"
    if [ -f "${hook_path}" ] || [ -L "${hook_path}" ]; then
        log_warning "Pre-push hook already exists at ${hook_path}"
        read -p "Overwrite existing hook? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping ${repo_name}"
            return 0
        fi
        rm -f "${hook_path}"
    fi

    # Create symlink to hook script (preferred for easy updates)
    if ln -s "${HOOK_SCRIPT}" "${hook_path}" 2>/dev/null; then
        log_success "Installed pre-push hook (symlink) in ${repo_name}"
    else
        # Fallback: copy script if symlink fails
        log_warning "Failed to create symlink, copying script instead"
        cp "${HOOK_SCRIPT}" "${hook_path}"
        chmod +x "${hook_path}"
        log_success "Installed pre-push hook (copy) in ${repo_name}"
    fi

    # Create .env file if it doesn't exist
    local env_file="${repo_path}/.env"
    if [ ! -f "${env_file}" ]; then
        log_info "Creating .env file with default configuration"
        cat > "${env_file}" <<EOF
# Git Hook Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:29092
KAFKA_ENABLE_LOGGING=true
GIT_HOOK_TIMEOUT=2
GIT_HOOK_DEBUG=false
EOF
        log_success "Created .env file: ${env_file}"
    else
        log_info "Using existing .env file: ${env_file}"
    fi

    # Test hook installation
    if [ -x "${hook_path}" ]; then
        log_success "Hook installed and executable: ${hook_path}"
    elif [ -f "${hook_path}" ]; then
        log_warning "Hook installed but not executable, fixing permissions"
        chmod +x "${hook_path}"
    else
        log_error "Hook installation failed: ${hook_path}"
        return 1
    fi

    return 0
}

# Uninstall hook from repository
uninstall_hook() {
    local repo_path="${1:-.}"
    local repo_name

    # Validate repository
    if ! is_git_repo "${repo_path}"; then
        log_error "Not a git repository: ${repo_path}"
        return 1
    fi

    repo_name=$(get_repo_name "${repo_path}")
    log_info "Uninstalling git hook from ${repo_name} (${repo_path})"

    # Remove pre-push hook
    local hook_path="${repo_path}/.git/hooks/pre-push"
    if [ -f "${hook_path}" ] || [ -L "${hook_path}" ]; then
        rm -f "${hook_path}"
        log_success "Uninstalled pre-push hook from ${repo_name}"
    else
        log_warning "No pre-push hook found in ${repo_name}"
    fi

    return 0
}

# Print configuration
print_config() {
    log_info "Git Hook Configuration:"
    echo "  KAFKA_BOOTSTRAP_SERVERS: ${KAFKA_BOOTSTRAP_SERVERS:-localhost:29092}"
    echo "  KAFKA_ENABLE_LOGGING: ${KAFKA_ENABLE_LOGGING:-true}"
    echo "  GIT_HOOK_TIMEOUT: ${GIT_HOOK_TIMEOUT:-2}"
    echo "  GIT_HOOK_DEBUG: ${GIT_HOOK_DEBUG:-false}"
    echo
}

# Print usage
print_usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [REPO_PATH...]

Install pre-push git hook that publishes file change events to Kafka.

Options:
  --uninstall       Uninstall hooks from repositories
  --help, -h        Show this help message

Arguments:
  REPO_PATH         Path to git repository (default: current directory)
                    Multiple paths can be specified for batch installation

Examples:
  # Install in current repository
  $0

  # Install in specific repository
  $0 /path/to/repo

  # Install in multiple repositories
  $0 /path/to/repo1 /path/to/repo2 /path/to/repo3

  # Uninstall from current repository
  $0 --uninstall

  # Uninstall from specific repositories
  $0 --uninstall /path/to/repo1 /path/to/repo2

Environment Variables:
  KAFKA_BOOTSTRAP_SERVERS   Kafka broker address (default: localhost:29092)
  KAFKA_ENABLE_LOGGING      Enable Kafka event logging (default: true)
  GIT_HOOK_TIMEOUT          Max execution time in seconds (default: 2)
  GIT_HOOK_DEBUG            Enable debug logging (default: false)

EOF
}

# Main function
main() {
    local uninstall=false
    local repo_paths=()

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --uninstall)
                uninstall=true
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                repo_paths+=("$1")
                shift
                ;;
        esac
    done

    # If no repository paths specified, use current directory
    if [ ${#repo_paths[@]} -eq 0 ]; then
        repo_paths=(".")
    fi

    # Print header
    echo
    log_info "Git Hook Installer - File Change Event Publisher"
    echo
    print_config

    # Check if hook script exists
    if [ ! -f "${HOOK_SCRIPT}" ]; then
        log_error "Hook script not found: ${HOOK_SCRIPT}"
        log_error "Please ensure you're running this script from the repository root"
        exit 1
    fi

    # Process each repository
    local success_count=0
    local failure_count=0

    for repo_path in "${repo_paths[@]}"; do
        if [ "$uninstall" = true ]; then
            if uninstall_hook "${repo_path}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
        else
            if install_hook "${repo_path}"; then
                ((success_count++))
            else
                ((failure_count++))
            fi
        fi
        echo
    done

    # Print summary
    log_info "Summary:"
    log_info "  Processed: ${#repo_paths[@]} repositories"
    log_success "  Successful: ${success_count}"
    if [ ${failure_count} -gt 0 ]; then
        log_error "  Failed: ${failure_count}"
    fi
    echo

    # Exit with appropriate code
    if [ ${failure_count} -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# Run main function
main "$@"
