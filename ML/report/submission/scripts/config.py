"""
config.py — Single source of paths, seeds, and CV settings for the entire pipeline.

Edit ONE thing here and every script picks it up. The teacher only needs to
change `_SUBMISSION_ROOT` if they relocate the folder.
"""

from __future__ import annotations

from pathlib import Path

# what: anchor every path off this file's location -> always correct, never a relative-path bug
# why: scripts can be run from anywhere (IDE, terminal, notebook) and still find data
# how: config.py lives in submission/scripts/, so the submission root is one level up
_SUBMISSION_ROOT = Path(__file__).resolve().parent.parent

# what: data folder bundled with the submission (consolidated_modeling_data.parquet lives here)
DATA_DIR = _SUBMISSION_ROOT / "data"

# what: where every script writes its outputs (created on demand, gitignored upstream)
OUTPUTS_DIR = _SUBMISSION_ROOT / "outputs"

# what: numerical-reproducibility seed used by every model factory and every np.random call
RANDOM_SEED = 42

# what: number of cross-validation folds used by 03_train_models, 04_calibration, 06_tuning
N_FOLDS = 5
