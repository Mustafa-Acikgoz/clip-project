import torch

# --- Project Paths ---
# Updated IMAGE_DIR to reflect where app.py downloads the dataset
IMAGE_DIR = "./flickr8k_images/Flicker8k_Dataset"
CAPTION_FILE = "data/captions.txt" # This path might need adjustment if captions.txt is also downloaded
MODEL_PATH = "clip_book_model.pth"

# --- Model Dimensions ---
IMAGE_EMBEDDING_DIM = 2048  
TEXT_EMBEDDING_DIM = 768  
PROJECTION_DIM = 256  

# --- Training Parameters ---
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3

# --- System ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")