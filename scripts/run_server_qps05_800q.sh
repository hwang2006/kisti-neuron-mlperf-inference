#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_server_case.sh" \
  "llama31-8b-server-qps0.5-800q.conf" \
  "server-qps0.5-800q"
