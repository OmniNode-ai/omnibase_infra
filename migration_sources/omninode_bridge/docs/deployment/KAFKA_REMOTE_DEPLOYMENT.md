# Kafka Remote Access Deployment Guide

## Overview

This guide covers deploying OmniNode Bridge with Kafka/Redpanda configured for remote access. The Kafka advertised listeners fix enables external clients (local development machines, integration tests, other services) to connect to Redpanda on the remote system.

**Remote System**: 192.168.86.200
**Kafka External Port**: 29102
**Internal Port**: 9092 (container-to-container)

## Quick Start

### New Deployment

If you're deploying for the first time:

```bash
# 1. Update username in migrate-to-remote.sh
REMOTE_USER="your_username"

# 2. Run migration (includes Kafka fix automatically)
./migrate-to-remote.sh

# 3. Verify Kafka connectivity
./verify-kafka-remote.sh
```

**That's it!** The migration script now automatically:
- Configures `KAFKA_ADVERTISED_HOST=192.168.86.200`
- Configures `KAFKA_ADVERTISED_PORT=29102`
- Syncs `docker-compose.remote-override.yml`
- Restarts Redpanda with correct configuration

### Existing Deployment

If you have an existing deployment without the Kafka fix:

```bash
# 1. Apply the fix
./deployment/scripts/apply-kafka-fix.sh

# 2. Verify connectivity
./verify-kafka-remote.sh
```

See [Manual Application](#manual-application-existing-deployment) section below for step-by-step instructions.

## Architecture

### Kafka Listener Configuration

Redpanda uses a two-listener architecture:

1. **Internal Listener** (`internal://omninode-bridge-redpanda:9092`)
   - Used by containers within Docker network
   - Service-to-service communication

2. **External Listener** (`external://192.168.86.200:29102`)
   - Used by external clients (local development, integration tests)
   - Advertised via environment variables

### Environment Variables

**remote.env**:
```bash
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=omninode-bridge-redpanda:9092  # Internal address

# Kafka Advertised Listeners for Remote Access
KAFKA_ADVERTISED_HOST=192.168.86.200  # Remote system IP
KAFKA_ADVERTISED_PORT=29102           # External port
```

**docker-compose.remote-override.yml**:
- Overrides Redpanda command with environment variable substitution
- Configures `--advertise-kafka-addr` with `${KAFKA_ADVERTISED_HOST}:${KAFKA_ADVERTISED_PORT}`
- Falls back to `localhost:29092` if variables not set

## Deployment Scenarios

### Scenario 1: Fresh Migration

**Prerequisites**:
- Local development machine with OmniNode Bridge running
- SSH access to 192.168.86.200
- Docker Desktop installed on remote system

**Steps**:

1. **Update migration script**:
   ```bash
   # Edit migrate-to-remote.sh
   REMOTE_USER="your_username"  # Change to your username
   ```

2. **Run migration**:
   ```bash
   ./migrate-to-remote.sh
   ```

3. **Wait for completion** (5-10 minutes):
   - Exports Docker images
   - Transfers files
   - Starts services
   - Verifies health

4. **Verify Kafka connectivity**:
   ```bash
   ./verify-kafka-remote.sh
   ```

   Expected output:
   ```
   ✅ Redpanda container is running
   ✅ Advertised address is correctly configured: 192.168.86.200:29102
   ✅ Port 29102 is accessible from this machine
   ✅ Environment variables configured
   ✅ docker-compose.remote-override.yml exists
   ```

5. **Test from local machine**:
   ```bash
   # List topics
   rpk topic list --brokers 192.168.86.200:29102

   # Test Python connection
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

### Scenario 2: Existing Deployment (Manual Application)

**Prerequisites**:
- OmniNode Bridge already deployed on 192.168.86.200
- SSH access to remote system
- Kafka currently advertising localhost

**Steps**:

1. **Verify current configuration**:
   ```bash
   ssh your_username@192.168.86.200
   cd ~/omninode_bridge
   docker exec omninode-bridge-redpanda rpk cluster info --brokers localhost:9092
   ```

   If output shows `localhost:29092`, continue with fix.

2. **Update remote.env**:
   ```bash
   ssh your_username@192.168.86.200
   cd ~/omninode_bridge

   # Backup existing configuration
   cp remote.env remote.env.backup

   # Add Kafka advertised listeners
   cat >> remote.env << 'EOF'

   # Kafka Advertised Listeners for Remote Access
   KAFKA_ADVERTISED_HOST=192.168.86.200
   KAFKA_ADVERTISED_PORT=29102
   EOF
   ```

3. **Transfer override file**:
   ```bash
   # On local machine
   scp docker-compose.remote-override.yml your_username@192.168.86.200:~/omninode_bridge/
   ```

4. **Restart Redpanda**:
   ```bash
   ssh your_username@192.168.86.200
   cd ~/omninode_bridge

   # Stop Redpanda
   docker-compose \
     -f deployment/docker-compose.yml \
     -f docker-compose.remote.yml \
     -f docker-compose.remote-override.yml \
     stop redpanda

   # Remove old container
   docker-compose \
     -f deployment/docker-compose.yml \
     -f docker-compose.remote.yml \
     -f docker-compose.remote-override.yml \
     rm -f redpanda

   # Start with new configuration
   docker-compose \
     -f deployment/docker-compose.yml \
     -f docker-compose.remote.yml \
     -f docker-compose.remote-override.yml \
     --env-file remote.env \
     up -d redpanda

   # Wait for Redpanda to start
   sleep 30
   ```

5. **Verify configuration**:
   ```bash
   docker exec omninode-bridge-redpanda rpk cluster info --brokers localhost:9092
   ```

   Expected output:
   ```
   BROKERS
   =======
   ID    HOST               PORT
   0*    192.168.86.200    29102
   ```

6. **Test external connectivity** (from local machine):
   ```bash
   ./verify-kafka-remote.sh
   ```

### Scenario 3: Rollback

If you need to rollback to localhost-only configuration:

1. **Remove Kafka advertised variables**:
   ```bash
   ssh your_username@192.168.86.200
   cd ~/omninode_bridge

   # Edit remote.env and remove:
   # KAFKA_ADVERTISED_HOST=192.168.86.200
   # KAFKA_ADVERTISED_PORT=29102
   ```

2. **Restart without override**:
   ```bash
   docker-compose \
     -f deployment/docker-compose.yml \
     -f docker-compose.remote.yml \
     --env-file remote.env \
     up -d redpanda
   ```

3. **Verify localhost configuration**:
   ```bash
   docker exec omninode-bridge-redpanda rpk cluster info --brokers localhost:9092
   # Should show: localhost:29092
   ```

## Verification

### Automated Verification

```bash
# Run verification script
./verify-kafka-remote.sh

# Example output:
# 1️⃣ Checking Redpanda container status...
# ✅ Redpanda container is running
#
# 2️⃣ Checking advertised Kafka address...
# ✅ Advertised address is correctly configured: 192.168.86.200:29102
#
# 3️⃣ Testing external Kafka connectivity...
# ✅ Port 29102 is accessible from this machine
#
# 4️⃣ Checking remote environment configuration...
# ✅ Environment variables configured
#
# 5️⃣ Checking docker-compose override file...
# ✅ docker-compose.remote-override.yml exists
```

### Manual Verification

**1. Check Advertised Address**:
```bash
ssh your_username@192.168.86.200 "docker exec omninode-bridge-redpanda rpk cluster info --brokers localhost:9092"
```

**Expected**:
```
BROKERS
=======
ID    HOST               PORT
0*    192.168.86.200    29102
```

**2. Test Port Connectivity**:
```bash
nc -zv 192.168.86.200 29102
```

**Expected**:
```
Connection to 192.168.86.200 port 29102 [tcp/*] succeeded!
```

**3. Test Python Client**:
```python
from aiokafka import AIOKafkaProducer
import asyncio

async def test_kafka():
    producer = AIOKafkaProducer(
        bootstrap_servers='192.168.86.200:29102',
        request_timeout_ms=10000
    )
    try:
        await producer.start()
        print("✅ Successfully connected to Kafka")
        await producer.stop()
    except Exception as e:
        print(f"❌ Connection failed: {e}")

asyncio.run(test_kafka())
```

**4. List Topics**:
```bash
rpk topic list --brokers 192.168.86.200:29102
```

## Troubleshooting

### Issue: Connection Refused

**Symptoms**:
```
Failed to connect to broker 192.168.86.200:29102
Connection refused
```

**Cause**: Firewall blocking port 29102

**Fix**:
```bash
# On remote system
ssh your_username@192.168.86.200
sudo ufw allow 29102/tcp
sudo ufw status

# Verify port is listening
sudo lsof -i :29102
```

### Issue: Still Advertising Localhost

**Symptoms**:
```bash
docker exec omninode-bridge-redpanda rpk cluster info --brokers localhost:9092
# Shows: localhost:29092 instead of 192.168.86.200:29102
```

**Cause**: Environment variables not passed to container

**Fix**:
```bash
# Verify environment variables are set
ssh your_username@192.168.86.200 "cd ~/omninode_bridge && grep KAFKA_ADVERTISED remote.env"

# Should show:
# KAFKA_ADVERTISED_HOST=192.168.86.200
# KAFKA_ADVERTISED_PORT=29102

# If missing, add them
ssh your_username@192.168.86.200 "cd ~/omninode_bridge && cat >> remote.env << 'EOF'

# Kafka Advertised Listeners for Remote Access
KAFKA_ADVERTISED_HOST=192.168.86.200
KAFKA_ADVERTISED_PORT=29102
EOF"

# Restart with override
ssh your_username@192.168.86.200 "cd ~/omninode_bridge && docker-compose -f deployment/docker-compose.yml -f docker-compose.remote.yml -f docker-compose.remote-override.yml --env-file remote.env up -d redpanda"
```

### Issue: Override File Not Applied

**Symptoms**:
Configuration shows localhost even after restart

**Cause**: docker-compose.remote-override.yml not included in command

**Fix**:
```bash
# Verify override file exists
ssh your_username@192.168.86.200 "ls -la ~/omninode_bridge/docker-compose.remote-override.yml"

# If missing, copy from local
scp docker-compose.remote-override.yml your_username@192.168.86.200:~/omninode_bridge/

# Restart with ALL compose files
ssh your_username@192.168.86.200 "cd ~/omninode_bridge && docker-compose \
  -f deployment/docker-compose.yml \
  -f docker-compose.remote.yml \
  -f docker-compose.remote-override.yml \
  --env-file remote.env \
  up -d redpanda"
```

### Issue: Integration Tests Failing

**Symptoms**:
```
aiokafka.errors.KafkaConnectionError: Unable to bootstrap from [('192.168.86.200', 29102)]
```

**Diagnostic**:
```bash
# 1. Verify advertised address
./verify-kafka-remote.sh

# 2. Test direct connection
telnet 192.168.86.200 29102

# 3. Check Redpanda logs
ssh your_username@192.168.86.200 "docker logs omninode-bridge-redpanda --tail=50"

# 4. Verify Kafka UI can connect
open http://192.168.86.200:8080
```

**Common Causes**:
1. Firewall blocking port
2. Redpanda not started with override
3. Environment variables not set
4. Network routing issue

## Performance Impact

**Negligible** - Advertised listeners configuration has no performance impact:
- No latency overhead
- No throughput reduction
- No memory increase

External connections may have slightly higher latency due to network routing vs localhost, but this is expected and minimal (typically <1ms).

## Security Considerations

### Exposed Ports

Port 29102 is now accessible from external networks.

**Firewall Configuration** (Recommended):
```bash
# Allow only from local network
sudo ufw allow from 192.168.86.0/24 to any port 29102

# Or allow from specific development IPs
sudo ufw allow from 192.168.86.105 to any port 29102
sudo ufw allow from 192.168.86.101 to any port 29102
```

### Production Deployment

For production environments:
1. **Use TLS encryption** for Kafka connections
2. **Enable SASL authentication** (username/password or SCRAM)
3. **Restrict IP access** via firewall rules
4. **Use VPN** for remote development access
5. **Monitor connections** via Kafka metrics

## Files Modified

The Kafka fix modifies/adds these files:

### Modified Files
- `remote.env` - Added `KAFKA_ADVERTISED_HOST` and `KAFKA_ADVERTISED_PORT`
- `migrate-to-remote.sh` - Added sync of override file
- `setup-remote.sh` - Added verification and override file usage
- `rebuild-service.sh` - Added override file to all docker-compose commands
- `docs/deployment/REMOTE_MIGRATION_GUIDE.md` - Added troubleshooting section

### New Files
- `docker-compose.remote-override.yml` - Kafka advertised listeners configuration
- `verify-kafka-remote.sh` - Verification script
- `docs/KAFKA_ADVERTISED_LISTENERS_FIX.md` - Technical documentation
- `docs/deployment/KAFKA_REMOTE_DEPLOYMENT.md` - This deployment guide

## Next Steps

After successful deployment:

1. **Update integration tests** to use remote Kafka:
   ```python
   KAFKA_BOOTSTRAP_SERVERS = os.getenv(
       "KAFKA_BOOTSTRAP_SERVERS",
       "192.168.86.200:29102"  # Use remote Kafka
   )
   ```

2. **Configure local development**:
   ```bash
   # In .env or environment
   export KAFKA_BOOTSTRAP_SERVERS=192.168.86.200:29102
   ```

3. **Document for team**:
   - Share remote Kafka address with team
   - Update README with connection details
   - Add troubleshooting notes for common issues

4. **Monitor performance**:
   ```bash
   # Check Redpanda metrics
   ssh your_username@192.168.86.200 "docker exec omninode-bridge-redpanda rpk cluster health --brokers localhost:9092"

   # View Kafka UI for metrics
   open http://192.168.86.200:8080
   ```

## References

- **Technical Details**: [docs/KAFKA_ADVERTISED_LISTENERS_FIX.md](../KAFKA_ADVERTISED_LISTENERS_FIX.md)
- **Migration Guide**: [docs/deployment/REMOTE_MIGRATION_GUIDE.md](./REMOTE_MIGRATION_GUIDE.md)
- **Redpanda Docs**: https://docs.redpanda.com/current/manage/kubernetes/networking/external-access/
- **Kafka Network Config**: https://kafka.apache.org/documentation/#brokerconfigs_advertised.listeners

## Support

For issues or questions:

1. Run `./verify-kafka-remote.sh` for diagnostic information
2. Check [Troubleshooting](#troubleshooting) section above
3. Review Redpanda logs: `ssh your_username@192.168.86.200 "docker logs omninode-bridge-redpanda"`
4. Verify network connectivity: `nc -zv 192.168.86.200 29102`
5. Check firewall rules: `ssh your_username@192.168.86.200 "sudo ufw status"`
