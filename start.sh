#!/bin/bash
set -a
source .env
set +a
source venv/bin/activate
python3 supervisor.py
