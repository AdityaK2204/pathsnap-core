"""Shared paths and pipeline constants."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Raw data: folder that contains train/ and student_test/.
# Override with the PATHSNAP_DATA environment variable.
_default_data = REPO_ROOT.parent / "Student folder" / "Student folder"
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", REPO_ROOT / "data"))
if not DATA_DIR.exists() and _default_data.exists():
    DATA_DIR = _default_data

TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "data_cache"   # derived feature tables (not committed)

SEED = 42

# Candidate generation
CAND_RADIUS_M = 60.0       # search radius around each GPS fix
CAND_MAX_K = 12            # keep at most this many arcs per fix
CAND_FALLBACK_K = 4        # if nothing is inside the radius, keep the nearest few
SAMPLE_SPACING_M = 10.0    # spacing of densified arc points used for the spatial index
CROP_MARGIN_M = 1500.0     # road network is cropped to the trip bounding box + margin

# Label derivation
LABEL_MAX_DIST_M = 50.0    # fixes further than this from their aligned route arc get no label

# Training subsample: keep at most this many fixes per training trip
MAX_FIXES_PER_TRIP = 1200

# HMM
GAP_BREAK_S = 120.0        # time gap that splits a trip into independent HMM segments
GAP_BREAK_M = 3000.0       # straight-line jump that splits a trip


def ensure_dirs():
    for d in (ARTIFACTS_DIR, RESULTS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
