#!/usr/bin/env bash
# Phase 1 — verify base decoupling: no static KG dependency, mock loop passes.
set -euo pipefail

echo "=== Phase 1: Base decoupling smoke test ==="
echo ""

# 1. Check Python imports resolve without errors
echo "[1/3] Verifying imports..."
python -c "
from data.kg_interface import SubGraph, MockSubGraphProvider, register_provider
from data.dataloader import ReplayBuffer, EpisodeCollector
from env.mdp_env import SemiMDPEnv, EnvConfig
from env.channel import awgn, rayleigh_fading
from models.transe import TransE
from models.lstm_path_memory import PathMemoryEncoder
from models.gnn_encoder import SubGraphEncoder
from models.generator import Generator
from models.discriminator import Discriminator
from agents.gail_agent import GAILAgent
from agents.slow_loop.mock_expert import MockExpert
from agents.slow_loop.intent_extractor import IntentExtractor
from agents.slow_loop.kg_pruner import prune_kg
print('  All imports OK.')
"

# 2. Assert no static file dependency in data pipeline
echo "[2/3] Checking for static file dependencies in data/..."
if grep -rn "open(" data/ --include="*.py" | grep -v "#" | grep -E "\.txt|\.csv" ; then
    echo "  FAIL: data/ modules contain static file I/O. Decouple required."
    exit 1
else
    echo "  OK: no static txt/csv dependency detected in data/."
fi

# 3. Run a 5-epoch debug training
echo "[3/3] Running 5-epoch debug training..."
python train.py --mode debug --epochs 5

echo ""
echo "=== Phase 1 smoke test PASSED ==="
