# config.py
import torch

# --- Project Paths ---
IMAGE_DIR = "data/Flicker8k_Dataset"
CAPTION_FILE = "data/captions.txt"
MODEL_PATH = "clip_book_model.pth"

# --- Model Dimensions ---
IMAGE_EMBEDDING_DIM = 2048  # ResNet50 output dimension
TEXT_EMBEDDING_DIM = 768    # DistilBERT output dimension
PROJECTION_DIM = 256        # Shared embedding space dimension

# --- Training Parameters ---
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3

# --- System ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")