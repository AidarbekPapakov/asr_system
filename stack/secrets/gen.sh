#!/bin/sh
set -e
echo "NEXTAUTH_SECRET=$(openssl rand -hex 32)"
echo "SALT=$(openssl rand -hex 16)"
echo "ENCRYPTION_KEY=$(openssl rand -hex 32)"
echo "PROJECT_PUBLIC_KEY=pk-lf-$(openssl rand -hex 16)"
echo "PROJECT_SECRET_KEY=sk-lf-$(openssl rand -hex 16)"
echo "USER_PASSWORD=$(openssl rand -hex 12)"
echo "LITELLM_MASTER_KEY=sk-$(openssl rand -hex 24)"
