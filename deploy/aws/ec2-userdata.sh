#!/bin/bash
# EC2 "User data" bootstrap script for the Nalanda RAG backend.
# Paste this into the EC2 launch wizard under Advanced details -> User data.
# Target AMI: Amazon Linux 2023.
#
# This installs Docker + Compose and clones the repo. It does NOT start the
# app automatically, because .env (with your API keys) still needs to be
# uploaded by hand after boot — see the deployment guide.

set -euxo pipefail

dnf update -y
dnf install -y docker git

systemctl enable --now docker
usermod -aG docker ec2-user

mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Amazon Linux 2023's docker package ships an old/no Buildx plugin;
# `docker compose build` requires buildx >= 0.17.0.
BUILDX_VERSION=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
curl -SL "https://github.com/docker/buildx/releases/download/${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.linux-amd64" \
  -o /usr/local/lib/docker/cli-plugins/docker-buildx
chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx

sudo -u ec2-user git clone https://github.com/flexish/nalanda_chatbot.git /home/ec2-user/app

echo "Bootstrap complete. Next steps (run as ec2-user over SSH):"
echo "  1. scp your local .env to /home/ec2-user/app/.env"
echo "  2. cd /home/ec2-user/app && docker compose up -d --build"
echo "  3. docker compose exec app python index.py --folder data"
