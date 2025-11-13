# Remote Migration Guide

## Overview

This guide covers migrating OmniNode Bridge containers from your local development machine to a remote system (192.168.86.200). The migration process uses two simple scripts that handle everything automatically.

## Quick Start

### Prerequisites

1. **Remote System Access**: SSH access to 192.168.86.200
2. **Docker Desktop**: Installed on remote system (will be updated automatically)
3. **Network Connectivity**: Both systems on same network (192.168.86.x)

### Migration Process

1. **Update the username** in `migrate-to-remote.sh`:
   ```bash
   REMOTE_USER="your_username"  # Change to your actual username
   ```

2. **Run the migration**:
   ```bash
   ./migrate-to-remote.sh
   ```

That's it! The script handles everything automatically.

## Scripts Overview

### 1. `migrate-to-remote.sh` (Local Machine)

**Purpose**: Handles the complete migration from your local machine.

**What it does**:
- ✅ Exports all Docker images
- ✅ Sets up SSH keys automatically
- ✅ Transfers everything to remote system
- ✅ Executes remote setup
- ✅ Verifies deployment
- ✅ Cleans up local files

**Usage**:
```bash
./migrate-to-remote.sh
```

### 2. `setup-remote.sh` (Remote System)

**Purpose**: Configures and starts everything on the remote system.

**What it does**:
- ✅ Updates Docker Desktop
- ✅ Imports all images
- ✅ Starts services in correct order
- ✅ Sets up firewall rules
- ✅ Creates management script
- ✅ Verifies everything is working

**Usage**: Automatically executed by the migration script.

### 3. `rebuild-service.sh` (Local Machine)

**Purpose**: Rebuilds and redeploys specific services after code changes.

**Usage**:
```bash
# Rebuild orchestrator
./rebuild-service.sh orchestrator

# Rebuild reducer
./rebuild-service.sh reducer

# Rebuild hook-receiver
./rebuild-service.sh hook-receiver

# See all available services
./rebuild-service.sh
```

## Service URLs

After successful migration, services will be available at:

### Application Services
- **Hook Receiver**: http://192.168.86.200:8001
- **Model Metrics**: http://192.168.86.200:8005
- **Orchestrator**: http://192.168.86.200:8060
- **Reducer**: http://192.168.86.200:8061
- **Registry**: http://192.168.86.200:8062
- **Metadata Stamping**: http://192.168.86.200:8057
- **OnexTree**: http://192.168.86.200:8058

### Infrastructure Services
- **Kafka/Redpanda**: `192.168.86.200:29102` (external access)
- **Consul UI**: http://192.168.86.200:28500
- **Vault UI**: http://192.168.86.200:8200
- **RedPanda UI**: http://192.168.86.200:8080

**Note**: Kafka external access is automatically configured via `docker-compose.remote-override.yml`

## Remote Management

### Using the Management Script

The remote system will have a `manage-bridge.sh` script for easy management:

```bash
# SSH to remote system
ssh your_username@192.168.86.200

# Navigate to project directory
cd ~/omninode_bridge

# Use management commands
./manage-bridge.sh status    # Show container status
./manage-bridge.sh logs      # Show logs
./manage-bridge.sh restart   # Restart services
./manage-bridge.sh stop      # Stop all services
./manage-bridge.sh start     # Start all services
./manage-bridge.sh update    # Update and restart
```

### Direct Docker Commands

```bash
# Check container status
ssh your_username@192.168.86.200 "cd ~/omninode_bridge && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml ps"

# View logs
ssh your_username@192.168.86.200 "cd ~/omninode_bridge && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml logs orchestrator"

# Restart specific service
ssh your_username@192.168.86.200 "cd ~/omninode_bridge && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml restart orchestrator"
```

## Development Workflow

### Making Code Changes

1. **Make changes** to your code locally
2. **Rebuild specific service**:
   ```bash
   ./rebuild-service.sh orchestrator
   ```
3. **Verify deployment** - The script automatically checks health

### Available Services for Rebuild

- `orchestrator` - Workflow coordination
- `reducer` - Metadata aggregation
- `hook-receiver` - Service lifecycle management
- `model-metrics` - AI Lab integration
- `metadata-stamping` - O.N.E. v0.1 Protocol support
- `onextree` - Project structure intelligence
- `registry` - Service registration

## Troubleshooting

### Common Issues

#### SSH Connection Failed
```bash
# Test SSH connection
ssh your_username@192.168.86.200 "echo 'Connection successful'"

# If failed, check:
# 1. Username is correct in migrate-to-remote.sh
# 2. SSH key is properly set up
# 3. Remote system is accessible
```

#### Docker Not Running on Remote
```bash
# SSH to remote and start Docker
ssh your_username@192.168.86.200 "open -a Docker"

# Wait for Docker to start, then retry migration
```

#### Service Health Check Failed
```bash
# Check container status
ssh your_username@192.168.86.200 "cd ~/omninode_bridge && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml ps"

# Check logs for specific service
ssh your_username@192.168.86.200 "cd ~/omninode_bridge && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml logs orchestrator"
```

#### Port Conflicts
```bash
# Check what's using the ports
ssh your_username@192.168.86.200 "lsof -i :8060"

# Stop conflicting services or change ports in docker-compose.remote.yml
```

#### Kafka Connectivity Issues

**Problem**: External clients (local development machines, integration tests) cannot connect to Kafka/Redpanda on remote system.

**Symptoms**:
- Connection refused when connecting to `192.168.86.200:29102`
- Broker advertises `localhost:29092` instead of remote IP
- Integration tests fail with "Unable to connect to Kafka"

**Root Cause**: Redpanda advertises `localhost` instead of the remote host IP address for external clients.

**Solution**:

The migration scripts now automatically configure Kafka advertised listeners for remote access via:
1. **remote.env** - Contains `KAFKA_ADVERTISED_HOST` and `KAFKA_ADVERTISED_PORT`
2. **docker-compose.remote-override.yml** - Configures Redpanda with environment variables

**Verification**:

```bash
# Run the verification script from local machine
./verify-kafka-remote.sh

# Or manually check advertised address
ssh your_username@192.168.86.200 "docker exec omninode-bridge-redpanda rpk cluster info --brokers localhost:9092"

# Expected output should show: 192.168.86.200:29102
```

**Manual Fix** (if verification fails):

1. **Update remote.env** on remote system:
   ```bash
   ssh your_username@192.168.86.200
   cd ~/omninode_bridge

   # Add Kafka configuration
   cat >> remote.env << 'EOF'

   # Kafka Advertised Listeners for Remote Access
   KAFKA_ADVERTISED_HOST=192.168.86.200
   KAFKA_ADVERTISED_PORT=29102
   EOF
   ```

2. **Restart Redpanda with override**:
   ```bash
   docker-compose \
     -f deployment/docker-compose.yml \
     -f docker-compose.remote.yml \
     -f docker-compose.remote-override.yml \
     --env-file remote.env \
     up -d redpanda
   ```

3. **Verify configuration**:
   ```bash
   docker exec omninode-bridge-redpanda rpk cluster info --brokers localhost:9092
   # Should show: HOST: 192.168.86.200, PORT: 29102
   ```

**Test Connection** from local development machine:

```bash
# Test with rpk (if installed)
rpk topic list --brokers 192.168.86.200:29102

# Test with Python
poetry run python -c "
from aiokafka import AIOKafkaProducer
import asyncio

async def test():
    producer = AIOKafkaProducer(bootstrap_servers='192.168.86.200:29102')
    await producer.start()
    print('✅ Connected successfully!')
    await producer.stop()

asyncio.run(test())
"
```

**Firewall Check**:

If connection still fails, ensure port 29102 is open:

```bash
# On remote system
sudo ufw allow 29102/tcp

# Check firewall status
sudo ufw status
```

**See Also**: [docs/KAFKA_ADVERTISED_LISTENERS_FIX.md](../KAFKA_ADVERTISED_LISTENERS_FIX.md) for detailed technical explanation.

### Rollback Strategy

If you need to rollback to local deployment:

```bash
# Stop remote containers
ssh your_username@192.168.86.200 "cd ~/omninode_bridge && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml down"

# Start local containers
docker-compose up -d
```

## Security Considerations

### Firewall Rules

The remote setup automatically configures firewall rules to:
- Allow access from local network (192.168.86.0/24)
- Block external access to services
- Allow SSH access

### Network Security

- Services are only accessible from local network
- No external internet access to service ports
- SSH key authentication required

### Environment Variables

Sensitive configuration is handled through environment variables:
- Database passwords
- API keys
- Service tokens

## Performance Considerations

### Resource Requirements

**Minimum Requirements**:
- 8GB RAM
- 4 CPU cores
- 50GB disk space

**Recommended**:
- 16GB RAM
- 8 CPU cores
- 100GB disk space

### Monitoring

Monitor resource usage:
```bash
# Check system resources
ssh your_username@192.168.86.200 "top -l 1"

# Check Docker resource usage
ssh your_username@192.168.86.200 "docker stats --no-stream"
```

## Advanced Configuration

### Custom Ports

To change service ports, edit `docker-compose.remote.yml`:

```yaml
services:
  orchestrator:
    ports:
      - "9060:8060"  # Change external port to 9060
```

### Environment Variables

To modify environment variables, edit `remote.env`:

```bash
# Change log level
LOG_LEVEL=debug

# Change database password
POSTGRES_PASSWORD=your_new_password
```

### Scaling Services

Scale specific services:

```bash
# Scale orchestrator to 3 instances
ssh your_username@192.168.86.200 "cd ~/omninode_bridge && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml up -d --scale orchestrator=3"
```

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review container logs for specific errors
3. Verify network connectivity and firewall rules
4. Check Docker Desktop is running and updated

## Migration Checklist

### Pre-Migration
- [ ] Update username in `migrate-to-remote.sh`
- [ ] Ensure remote system is accessible
- [ ] Verify local containers are running
- [ ] Check available disk space on remote system

### Migration Execution
- [ ] Run `./migrate-to-remote.sh`
- [ ] Enter SSH password when prompted
- [ ] Wait for all services to start
- [ ] Verify all service URLs are accessible

### Post-Migration
- [ ] Test all service endpoints
- [ ] Verify container health
- [ ] Verify Kafka external connectivity with `./verify-kafka-remote.sh`
- [ ] Test rebuild workflow with `./rebuild-service.sh`
- [ ] Document any custom configurations
