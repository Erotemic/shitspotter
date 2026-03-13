#!/bin/bash
set -euo pipefail

# shellcheck source=experiments/foundation_detseg_v3/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

python -m pytest -q \
    "$FOUNDATION_V3_ROOT_DIR/tests/test_import.py" \
    "$FOUNDATION_V3_ROOT_DIR/shitspotter/algo_foundation_v3/tests/test_foundation_v3.py"
