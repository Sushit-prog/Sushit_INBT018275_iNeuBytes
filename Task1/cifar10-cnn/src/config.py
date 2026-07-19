
from pathlib import Path


SEED = 42


# Keras' cifar10.load_data() gives 50,000 train / 10,000 test images.
# We carve a fixed validation set out of the 50,000 train images.
VAL_FRACTION = 0.10  # -> 45,000 train / 5,000 val / 10,000 test

# Training budget 
BATCH_SIZE = 64
EPOCHS = 20  

# If Part B needs more epochs, set this explicitly (fails loudly otherwise).
JUSTIFY_EPOCH_CHANGE = None 
CUSTOM_EPOCHS = EPOCHS  

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SHAPE = (32, 32, 3)

# Success thresholds from the task spec
TRADITIONAL_MIN_TEST_ACC = 0.70
CUSTOM_MIN_IMPROVEMENT_PP = 3.0  # percentage points over traditional CNN

# Paths 
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"

for _dir in (FIGURES_DIR, METRICS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def resolve_custom_epochs() -> int:
    """Guards against silently unfair comparisons between Part A and B."""
    if CUSTOM_EPOCHS != EPOCHS and not JUSTIFY_EPOCH_CHANGE:
        raise ValueError(
            "CUSTOM_EPOCHS differs from EPOCHS but JUSTIFY_EPOCH_CHANGE is "
            "empty. Set a justification string in config.py — the task spec "
            "requires the epoch budget change to be stated and justified."
        )
    return CUSTOM_EPOCHS
