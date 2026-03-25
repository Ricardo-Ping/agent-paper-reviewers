#!/usr/bin/env bash
set -euo pipefail
python -m reviewers_sim.cli run --input "$1" --output-dir "${2:-runs}"
