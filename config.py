# config.py

# Hugging Face Dataset Details
REPO_ID = "mustafa2ak/Flickr8k-Images"
IMAGE_DIR_NAME = "Flicker8k_Dataset"
CAPTION_FILE_NAME = "captions.txt"

# Model and Training Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 3
LR = 1e-3
SAVED_MODEL_PATH = "flickr8k_clip_model.pth"

# Model Dimensions
IMAGE_EMBEDDING_DIM = 2048  # From ResNet50
TEXT_EMBEDDING_DIM = 768   # From DistilBERT
PROJECTION_DIM = 256