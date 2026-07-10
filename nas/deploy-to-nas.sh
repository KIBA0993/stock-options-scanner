#!/bin/bash
# deploy-to-nas.sh — sync ~/trading to Synology DS920+ and start the scanner container
#
# Prerequisites on NAS:
#   1. Control Panel → Terminal & SNMP → Enable SSH
#   2. Package Center → Container Manager (Docker)
#   3. User has permission to run docker (or use sudo in REMOTE_DOCKER)
#
# Usage:
#   cp nas/nas.env.example nas/nas.env   # edit NAS_HOST / NAS_USER
#   ./nas/deploy-to-nas.sh
#   ./nas/deploy-to-nas.sh --test-scan   # run one scan after deploy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRADING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$SCRIPT_DIR/nas.env"
TEST_SCAN=false

for arg in "$@"; do
  [[ "$arg" == "--test-scan" ]] && TEST_SCAN=true
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE"
  echo "  cp nas/nas.env.example nas/nas.env"
  echo "  # then set NAS_HOST, NAS_USER, NAS_PATH"
  exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

: "${NAS_HOST:?Set NAS_HOST in nas.env}"
: "${NAS_USER:?Set NAS_USER in nas.env}"
: "${NAS_PATH:=/volume1/docker/trading}"
: "${NAS_SSH_KEY:=$HOME/.ssh/id_ed25519_nas}"

REMOTE="${NAS_USER}@${NAS_HOST}"
NAS_DOCKER_PATH="${NAS_DOCKER_PATH:-/usr/local/bin/docker}"
NAS_COMPOSE_PATH="${NAS_COMPOSE_PATH:-/usr/local/bin/docker-compose}"
NAS_DOCKER_ENV="env PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SSH_KEY="${NAS_SSH_KEY/#\~/$HOME}"
SSH_OPTS=(-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
[[ -f "$SSH_KEY" ]] && SSH_OPTS+=(-i "$SSH_KEY")
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

ssh_nas() {
  ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"
}

docker_nas() {
  if [[ -n "${NAS_SUDO_PASSWORD:-}" ]]; then
    ssh_nas "echo '$NAS_SUDO_PASSWORD' | sudo -S sh -c 'cd \"$NAS_PATH/nas\" && $NAS_DOCKER_ENV $NAS_COMPOSE_PATH $*'"
  else
    ssh_nas "sudo -n sh -c 'cd \"$NAS_PATH/nas\" && $NAS_DOCKER_ENV $NAS_COMPOSE_PATH $*'" 2>/dev/null || {
      echo "ERROR: docker needs sudo on Synology. Set NAS_SUDO_PASSWORD in nas.env for deploy."
      exit 1
    }
  fi
}

docker_exec_nas() {
  if [[ -n "${NAS_SUDO_PASSWORD:-}" ]]; then
    ssh_nas "echo '$NAS_SUDO_PASSWORD' | sudo -S $NAS_DOCKER_ENV $NAS_DOCKER_PATH exec trading-scanner $*"
  else
    ssh_nas "sudo -n $NAS_DOCKER_ENV $NAS_DOCKER_PATH exec trading-scanner $*" 2>/dev/null || {
      echo "ERROR: docker exec needs sudo on Synology."
      exit 1
    }
  fi
}

rsync_nas() {
  # Synology DSM often disables rsync/scp SFTP subsystem; tar-over-SSH works.
  local dest_path="$1"
  shift
  tar czf - "$@" | ssh_nas "mkdir -p '$dest_path' && cd '$dest_path' && tar xzf -"
}

sync_trading_to_nas() {
  echo "==> Syncing trading project (tar over SSH) ..."
  (
    cd "$TRADING_DIR"
    tar czf - \
      --exclude='.git' \
      --exclude='__pycache__' \
      --exclude='.pytest_cache' \
      --exclude='.python-version' \
      --exclude='logs' \
      --exclude='data/archive' \
      --exclude='creators/*/posts_raw.txt' \
      --exclude='creators/*/amendments' \
      .
  ) | ssh_nas "mkdir -p '$NAS_PATH' && cd '$NAS_PATH' && tar xzf -"
}

ensure_scripts_executable() {
  echo "==> Ensuring NAS scripts are executable ..."
  ssh_nas "chmod +x '$NAS_PATH/nas/scripts/'*.sh 2>/dev/null || true"
}

ensure_ssh_access() {
  if ssh_nas "echo ok" &>/dev/null; then
    return 0
  fi
  if [[ -n "${NAS_PASSWORD:-}" ]] && [[ -f "$SCRIPT_DIR/install-nas-ssh-key.exp" ]]; then
    echo "==> Installing SSH key on NAS (one-time) ..."
    NAS_PASSWORD="$NAS_PASSWORD" expect "$SCRIPT_DIR/install-nas-ssh-key.exp" \
      "$REMOTE" "${SSH_KEY}.pub"
    ssh_nas "echo ok"
    return $?
  fi
  echo "ERROR: Cannot SSH to $REMOTE"
  echo "  Add this key in DSM, or set NAS_PASSWORD in nas/nas.env for one-time install:"
  echo "  $(cat "${SSH_KEY}.pub" 2>/dev/null || echo '(run: ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_nas -N \"\")')"
  exit 1
}

ensure_ssh_access

echo "==> Checking SSH to $REMOTE ..."
ssh_nas "echo NAS reachable: \$(hostname)"

echo "==> Creating remote directory $NAS_PATH ..."
ssh_nas "mkdir -p '$NAS_PATH'"

sync_trading_to_nas

ensure_scripts_executable

if ! ssh_nas "test -f '$NAS_PATH/config.json'"; then
  echo "ERROR: config.json not found on NAS after sync."
  echo "Ensure ~/trading/config.json exists on your Mac (it is gitignored)."
  exit 1
fi

echo "==> Building and starting container ..."
docker_nas up -d --build

echo "==> Container status:"
docker_nas ps

if $TEST_SCAN; then
  echo "==> Running test scan on NAS ..."
  docker_exec_nas /scripts/morning_digest_runner.sh
  echo "Check logs: ssh ${SSH_OPTS[*]} $REMOTE 'tail -50 $NAS_PATH/logs/morning_digest.log'"
fi

cat <<EOF

Deploy complete.

NAS schedules (ET, Mon–Fri):
  9:45 AM  morning digest (always emails; call/put contracts included)
  12:45 PM midday scan (email on new alerts; contracts included)

Useful commands:
  ssh $REMOTE "tail -f $NAS_PATH/logs/scan.log"
  ssh $REMOTE "tail -f $NAS_PATH/logs/morning_digest.log"

Disable Mac launchd jobs (avoid duplicate scans):
  launchctl unload ~/Library/LaunchAgents/com.trading.scan-midday.plist
  launchctl unload ~/Library/LaunchAgents/com.trading.scan-afternoon.plist
  launchctl unload ~/Library/LaunchAgents/com.trading.morning-digest.plist
EOF
