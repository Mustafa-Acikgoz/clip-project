import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer
from huggingface_hub import snapshot_download
import os
import glob
from tqdm import tqdm

# --- Custom Modules ---
import config
from inference_model import CLIPModel

# --- 1. Initial Setup: Load Model and Tokenizer ---
print("Starting application setup...")
device = config.DEVICE

# Load the CLIP model's structure
model = CLIPModel(
    image_embedding_dim=config.IMAGE_EMBEDDING_DIM,
    text_embedding_dim=config.TEXT_EMBEDDING_DIM,
    projection_dim=config.PROJECTION_DIM
).to(device)

# --- CRITICAL STEP (Corrected) ---
# Load the state dictionary with `strict=False`.
# This allows the model to load only the weights present in the file (e.g., your trained
# projection heads) and ignore the missing ones (e.g., the base ResNet and DistilBERT weights,
# which are already pre-loaded by the model class itself).
try:
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device), strict=False)
    model.eval()
    print("CLIP Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the text tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print("Tokenizer loaded successfully.")


# --- 2. Data Handling: Download and Pre-process Images ---
DATASET_REPO_ID = "mustafa2ak/Flickr8k-Images"
IMAGE_STORAGE_PATH = "./flickr8k_images"

print(f"Downloading image dataset from {DATASET_REPO_ID}...")
snapshot_download(
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    local_dir=IMAGE_STORAGE_PATH,
    local_dir_use_symlinks=False
)
print("Image dataset download complete.")

# Get a list of all image file paths
all_image_paths = glob.glob(os.path.join(IMAGE_STORAGE_PATH, "Flicker8k_Dataset", "*.jpg"))

# Use a smaller subset of images to prevent timeouts and for faster testing.
# You can increase this value after confirming the app works.
NUM_IMAGES_TO_PROCESS = 1000
all_image_paths = all_image_paths[:NUM_IMAGES_TO_PROCESS]
print(f"Found {len(all_image_paths)} total images. Using a subset of {NUM_IMAGES_TO_PROCESS} to prevent timeout.")

# Define the image preprocessing pipeline
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def precompute_image_embeddings(image_paths, model, transform, device):
    """Processes all images and computes their final embeddings for fast searching."""
    print("Pre-computing image embeddings... This may take a minute.")
    all_embeddings = []
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Processing Images"):
            try:
                image = Image.open(path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Pass image_features to the model to get the embedding
                embedding, _ = model(image_features=image_tensor)
                
                all_embeddings.append(embedding)
            except Exception as e:
                print(f"Warning: Could not process image {path}. Error: {e}")
                continue
    return torch.cat(all_embeddings, dim=0)

# Pre-compute all image embeddings and store them in memory
if model and all_image_paths:
    image_embeddings_precomputed = precompute_image_embeddings(all_image_paths, model, image_transform, device)
    # Normalize the embeddings once for faster similarity calculation
    image_embeddings_precomputed = F.normalize(image_embeddings_precomputed, p=2, dim=-1)
    print("Image embeddings pre-computed and stored.")
else:
    image_embeddings_precomputed = None
    print("Skipping embedding pre-computation due to missing model or images.")


# --- 3. The Main Gradio Function for Text-to-Image Search ---
def find_image_from_text(text_query):
    """Takes a text query and finds the best matching image."""
    if not text_query:
        return None, "Please enter a text query."
    if image_embeddings_precomputed is None:
        return None, "Error: Image embeddings are not available. Check logs for errors."

    print(f"Searching for text: '{text_query}'")
    with torch.no_grad():
        # 1. Process the text query
        text_inputs = tokenizer([text_query], padding=True, truncation=True, return_tensors="pt").to(device)
        
        # 2. Get the projected text embedding from the model.
        _, text_embedding = model(
            text_input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask']
        )
        
        # 3. Normalize the text embedding
        text_embedding_norm = F.normalize(text_embedding, p=2, dim=-1)

        # 4. Calculate similarity against all pre-computed image embeddings
        similarity_scores = (text_embedding_norm @ image_embeddings_precomputed.T).squeeze(0)

        # 5. Find the index of the image with the highest score
        best_image_index = similarity_scores.argmax().item()
        best_image_path = all_image_paths[best_image_index]
        best_score = similarity_scores[best_image_index].item()

        print(f"Found best match: {best_image_path} with score {best_score:.4f}")

        return best_image_path, f"Best match with score: {best_score:.4f}"


# --- 4. Create and Launch the Gradio Interface ---
iface = gr.Interface(
    fn=find_image_from_text,
    inputs=gr.Textbox(lines=2, label="Text Query", placeholder="Enter text to find a matching image..."),
    outputs=[
        gr.Image(type="filepath", label="Best Matching Image"),
        gr.Textbox(label="Result Details")
    ],
    title="🖼️ Text-to-Image Search with CLIP",
    description="Enter a text description to search for the most relevant image in the Flickr8k dataset. The app uses a pre-trained CLIP-like model to find the best match.",
    allow_flagging="never"
)

iface.launch()