#!/usr/bin/env bash
# Provision the Open Problems label_projection benchmark box (run via SSM send-command).
# Idempotent: safe to re-send if a step fails.
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
cd /home/ubuntu

# --- toolchain -------------------------------------------------------------------
# Each tool is checked independently: nesting them under one guard means a re-send
# after a partial failure skips the steps that never ran.
if ! command -v docker >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq docker.io awscli git unzip curl
  systemctl enable --now docker
  usermod -aG docker ubuntu
fi

# Nextflow needs Java 17+; Ubuntu 22.04's `default-jre` is Java 11, which it rejects.
if ! java -version 2>&1 | grep -qE '"(1[7-9]|2[0-9])'; then
  apt-get install -y -qq openjdk-17-jre-headless
fi
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"

if ! command -v nextflow >/dev/null; then
  curl -s https://get.nextflow.io | bash
  mv nextflow /usr/local/bin/ && chmod +x /usr/local/bin/nextflow
fi
if ! command -v viash >/dev/null; then
  curl -fsSL https://github.com/viash-io/viash/releases/latest/download/viash -o /usr/local/bin/viash
  chmod +x /usr/local/bin/viash
fi

# --- the task repo ---------------------------------------------------------------
if [ ! -d task_label_projection ]; then
  git clone -q https://github.com/openproblems-bio/task_label_projection.git
fi
chown -R ubuntu:ubuntu /home/ubuntu

# Make Java 17 the default for later (non-login) SSM shells too.
echo "export JAVA_HOME=$JAVA_HOME" > /etc/profile.d/java17.sh
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> /etc/profile.d/java17.sh

echo "SETUP_OK"
java -version 2>&1 | head -1; nextflow -v; viash --version; docker --version
echo "cores=$(nproc)"; free -g | head -2; df -h / | tail -1
