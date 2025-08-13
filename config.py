# config.py

# --- Hugging Face Repository Details ---
# The repo where your dataset, images, and trained model are stored.
REPO_ID = "mustafa2ak/Flickr8k-Images"
IMAGE_DIR_NAME = "Flicker8k_Dataset"
CAPTION_FILE_NAME = "captions.txt"

# --- Training Configuration ---
BATCH_SIZE = 32
NUM_EPOCHS = 3
LR = 1e-3 # Learning Rate

# --- Model and File Paths ---
# The name for your saved model file after training.
SAVED_MODEL_PATH = "flickr8k_clip_model.pth"

# --- Model Dimensions ---
# These must match the encoders you are using.
IMAGE_EMBEDDING_DIM = 2048  # From ResNet50
TEXT_EMBEDDING_DIM = 768    # From DistilBERT
PROJECTION_DIM = 256        # The size of the shared space