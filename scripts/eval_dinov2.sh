#!/usr/bin/env bash
set -euo pipefail
set -a; [[ -f .env ]] && source .env; set +a
uv run python -m nanodino.evaluate --checkpoint outputs/dinov2/teacher_backbone.pt --data-root ./data
