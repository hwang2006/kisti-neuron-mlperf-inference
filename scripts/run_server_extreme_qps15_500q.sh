#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_server_case.sh" \
  "llama31-8b-server-extreme-qps1.5-500q.conf" \
  "server-extreme-qps1.5-500q"
