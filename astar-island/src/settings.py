"""
Astar Island — Configuration & Constants
"""
import os

# API
API_BASE = "https://api.ainm.no/astar-island"
TOKEN = os.environ.get("AINM_TOKEN", "")

# Grid
MAP_H = 40
MAP_W = 40
NUM_CLASSES = 6
SIM_YEARS = 50
VIEWPORT_SIZE = 15

# Grid value → prediction class mapping
GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
CLASS_NAMES = ["Empty/Ocean/Plains", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

# Observation budget
MAX_QUERIES = 50

# Data paths
from pathlib import Path
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
