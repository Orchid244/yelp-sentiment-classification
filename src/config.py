import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
SPLIT_DIR = os.path.join(OUTPUT_DIR, "splits")

BASELINE_DIR = os.path.join(RESULT_DIR, "baseline")
TRANSFORMER_MODEL_DIR = os.path.join(MODEL_DIR, "transformer")
TRANSFORMER_RESULT_DIR = os.path.join(RESULT_DIR, "transformer")

DATA_PATH = os.path.join(DATA_DIR, "Yelp Restaurant Reviews.csv")

# Columns
TEXT_COL = "Review Text"
RATING_COL = "Rating"

# Labels
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Split
RANDOM_SEED = 42
TEST_SIZE = 0.10
VAL_SIZE = 0.10

# Baseline
TFIDF_MAX_FEATURES = 30000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

# Transformer
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
RUN_FROZEN_ENCODER = True
RUN_FULL_FINETUNE = True