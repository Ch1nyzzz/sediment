#!/bin/bash
cd "$(dirname "$0")/.."
pgrep -f "bash scripts/server_watchdo[g].sh" >/dev/null && { echo "watchdog already running"; exit 0; }
nohup bash scripts/server_watchdog.sh > results/watchdog.log 2>&1 < /dev/null &
disown
sleep 3; pgrep -af "bash scripts/server_watchdo[g].sh" | cut -c1-80; ls -la results/watchdog.log
