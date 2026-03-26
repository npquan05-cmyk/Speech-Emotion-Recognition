ROOT_PATH = "/kaggle/input/datasets/uwrfkaggler/ravdess-emotional-speech-audio"

MODEL_CHECKPOINT = "facebook/wav2vec2-base-960h"

TEST_SIZE = 0.2
SEED = 42

MAX_DURATION = 5  # seconds
SAMPLING_RATE = 16000

OUTPUT_DIR = "./wav2vec2-ser"
BEST_MODEL_DIR = "./best_model"

NUM_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
