#!/usr/bin/env bash
set -euo pipefail
python -m agent_paper_reviewers.cli run --input "$1" --output-dir "${2:-runs}"

