RANDOM_STATE = 42

TARGET_COL = "Price"

# Filtering rules (you can tune)
MAX_PRICE = 200000
MAX_KM = 200000
MIN_YEAR = 2000

NEAR_NEW_KM = 10000
RET_CLIP_LOW = 0.02
RET_CLIP_HIGH = 1.30

NUM_COLS = [
    "Age",
    "log_km",
    "FuelConsumption",
    "CylindersinEngine",
    "Seats",
    "age_kilometer_interaction",
]

CAT_COLS = [
    "Brand",
    "Model",
    "UsedOrNew",
    "DriveType",
    "BodyType",
    "Transmission",
    "FuelType",
]

MODEL_GROUP_COL = "BrandModelGroup"  # created during feature build
