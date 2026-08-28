"""
TURA-GeoSync Configuration Module
Provides centralized relative paths and system configuration constants.
"""

from pathlib import Path

# Base Repository Directory (Dynamic Relative Path)
BASE_DIR = Path(__file__).resolve().parent

# Data Directories
DATA_DIR = BASE_DIR / "grace_data"

# Default File Paths
DEFAULT_GRACE_NC_FILENAME = "GRCTellus.JPL.200204_202606.GLO.RL06.3M.MSCNv04CRI.nc"
DEFAULT_GRACE_NC_PATH = DATA_DIR / DEFAULT_GRACE_NC_FILENAME

# Remote Data Sources
IERS_EOP_URL = "https://hpiers.obspm.fr/iers/eop/eopc04/eopc04.1962-now"

# Region Definitions (Bounding Boxes)
REGIONS = {
    "himalayas": {
        "name": "Himalayas and Tibetan Plateau",
        "lat_min": 25,
        "lat_max": 40,
        "lon_min": 70,
        "lon_max": 100
    }
}

# Machine Learning Default Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
RF_N_ESTIMATORS = 100
