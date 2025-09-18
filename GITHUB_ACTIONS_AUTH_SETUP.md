# GitHub Actions Authentication Setup for Private Dependencies

## Issue Summary

The CI/CD pipeline is failing because GitHub Actions cannot access private repositories (`omnibase_spi` and `omnibase_core`) with the default `GITHUB_TOKEN`. This token only has access to the current repository.

## Current Failure

```
Failed to clone https://github.com/OmniNode-ai/omnibase_spi.git,
check your git configuration and permissions for this repository.
```

## Root Cause

- **Default GITHUB_TOKEN**: Limited to current repository only
- **Private Dependencies**: Require cross-repository access
- **Poetry Installation**: Needs to clone private Git repositories

## Solutions

### Option 1: Personal Access Token (PAT) - RECOMMENDED

1. **Create PAT**:
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Create a new token with `repo` scope (full repository access)
   - Copy the token value

2. **Add Repository Secret**:
   - Go to repository Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `ONEX_PAT_TOKEN`
   - Value: The PAT token created above

3. **Update Workflow**:
   ```yaml
   - name: Configure Git for private repositories
     env:
       ONEX_PAT_TOKEN: ${{ secrets.ONEX_PAT_TOKEN }}
     run: |
       git config --global url."https://x-access-token:${ONEX_PAT_TOKEN}@github.com/".insteadOf "https://github.com/"
   ```

### Option 2: Deploy Keys (More Secure)

1. **Generate SSH Key Pair**:
   ```bash
   ssh-keygen -t ed25519 -f deploy_key -N ""
   ```

2. **Add Deploy Keys**:
   - Add public key to each private repository (Settings → Deploy keys)
   - Add private key as repository secret: `DEPLOY_PRIVATE_KEY`

3. **Update Workflow** (more complex SSH setup required)

### Option 3: Organization App Token (Enterprise)

- Use GitHub App with cross-repository permissions
- Requires organization-level setup

## Current Temporary Solution

The workflow now:
1. **Attempts private repository access** with current token
2. **Falls back to public dependencies only** if access fails
3. **Provides clear configuration instructions** in CI logs
4. **Continues with available dependencies** to test what's possible

## Configuration Instructions

### For Repository Administrators

1. **Choose Option 1 (PAT)** for simplest setup
2. **Create PAT** with `repo` scope from a user with access to all private repos
3. **Add secret** `ONEX_PAT_TOKEN` to this repository
4. **Update workflow** to use the new token (instructions in logs)

### Verification

After configuration, check that:
- [ ] CI can clone `omnibase_spi` repository
- [ ] CI can clone `omnibase_core` repository
- [ ] Poetry can install all dependencies
- [ ] Tests can import all required modules

## Security Notes

- **PAT Scope**: Use minimal required scope (`repo` for private repos)
- **PAT Rotation**: Regularly rotate the PAT token
- **Access Review**: Periodically review who has access to the PAT
- **Alternative**: Consider GitHub Apps for organization-wide solutions

## Next Steps

1. **Configure authentication** using Option 1 (PAT)
2. **Test CI pipeline** with new authentication
3. **Remove `continue-on-error`** flags once dependencies install successfully
4. **Enable full test suite** including type checking and testing

This setup will enable the full CI/CD pipeline to validate the domain model implementations properly.