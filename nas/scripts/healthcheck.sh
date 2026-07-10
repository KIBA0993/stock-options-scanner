#!/bin/bash
# Verify outbound internet before scans (stable NAS ethernet + reliable DNS).

set -euo pipefail

check() {
  local url="$1"
  curl -fsS --max-time 15 -o /dev/null "$url"
}

check "https://www.google.com/generate_204" \
  || check "https://1.1.1.1/cdn-cgi/trace"
