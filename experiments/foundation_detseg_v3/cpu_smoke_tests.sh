#!/bin/bash
set -euo pipefail

python -m pytest -q \
    tests/test_import.py \
    shitspotter/algo_foundation_v3/tests/test_foundation_v3.py
