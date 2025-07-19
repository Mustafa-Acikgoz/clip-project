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
# These imports assume your config.py and model files are in the same directory
import config
from inference_model import CLIPModel

# --- 1. Initial Setup: Load Model and Tokenizer (runs once on startup) ---
print("Starting application setup...")
device = config.DEVICE

# Load the CLIP model's structure
model = CLIPModel(
    image_embedding_dim=config.IMAGE_EMBEDDING_DIM,
    text_embedding_dim=config.TEXT_EMBEDDING_DIM,
    projection_dim=config.PROJECTION_DIM
).to(device)

# Load your trained model weights (.pth file)
try:
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()
    print("CLIP Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

# Load the text tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print("Tokenizer loaded successfully.")


# --- 2. Data Handling: Download and Pre-process Images (runs once on startup) ---
# This is the key section that connects your app to your image dataset.

# Define the dataset repository on the Hugging Face Hub
DATASET_REPO_ID = "mustafa2ak/Flickr8k-Images" 
# Define the local folder where the images will be stored inside the Space
IMAGE_STORAGE_PATH = "./flickr8k_images"

print(f"Downloading image dataset from {DATASET_REPO_ID}...")
# Use snapshot_download for a fast, server-to-server transfer
snapshot_download(
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    local_dir=IMAGE_STORAGE_PATH,
    local_dir_use_symlinks=False # Important for compatibility
)
print("Image dataset download complete.")

# Get a list of all image file paths from the downloaded folder
# It looks for all .jpg files inside the 'images' subfolder you created
all_image_paths = glob.glob(os.path.join(IMAGE_STORAGE_PATH, "images", "*.jpg"))
print(f"Found {len(all_image_paths)} images.")

# Define the image preprocessing pipeline
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def precompute_image_embeddings(image_paths, model, transform, device):
    """
    Processes all images and computes their embeddings for fast searching.
    This is a crucial optimization.
    """
    print("Pre-computing image embeddings... This may take a few minutes.")
    all_embeddings = []
    # torch.no_grad() disables gradient calculation, making this much faster
    with torch.no_grad():
        # tqdm creates a progress bar in your logs
        for path in tqdm(image_paths, desc="Processing Images"):
            try:
                image = Image.open(path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)
                # Pass the image through the model's image encoder part
                embedding = model.image_encoder(image_tensor)
                all_embeddings.append(embedding)
            except Exception as e:
                print(f"Warning: Could not process image {path}. Error: {e}")
                continue
    # Combine the list of individual tensors into one large tensor
    return torch.cat(all_embeddings, dim=0)

# Pre-compute all image embeddings and store them in memory
if model and all_image_paths:
    image_embeddings_precomputed = precompute_image_embeddings(all_image_paths, model, image_transform, device)
    # Normalize the embeddings once for faster similarity calculation later
    image_embeddings_precomputed = F.normalize(image_embeddings_precomputed, p=2, dim=-1)
    print("Image embeddings pre-computed and stored.")
else:
    image_embeddings_precomputed = None
    print("Skipping embedding pre-computation due to missing model or images.")


# --- 3. The Main Gradio Function for Text-to-Image Search ---
def find_image_from_text(text_query):
    """
    Takes a text query and finds the best matching image from the pre-computed embeddings.
    """
    if not text_query:
        return None, "Please enter a text query."
    if image_embeddings_precomputed is None:
        return None, "Error: Image embeddings are not available. Check logs for errors."

    print(f"Searching for text: '{text_query}'")
    with torch.no_grad():
        # 1. Process the text query into tokens and get its embedding
        text_inputs = tokenizer([text_query], padding=True, truncation=True, return_tensors="pt").to(device)
        text_embedding = model.text_encoder(
            input_ids=text_inputs['input_ids'], 
            attention_mask=text_inputs['attention_mask']
        )
        # 2. Normalize the text embedding
        text_embedding_norm = F.normalize(text_embedding, p=2, dim=-1)

        # 3. Calculate similarity against all pre-computed image embeddings
        # This is a fast matrix multiplication: (1, 512) @ (512, N_images) -> (1, N_images)
        similarity_scores = (text_embedding_norm @ image_embeddings_precomputed.T).squeeze(0)

        # 4. Find the index of the image with the highest score
        best_image_index = similarity_scores.argmax().item()
        
        # 5. Get the file path of the best image
        best_image_path = all_image_paths[best_image_index]
        best_score = similarity_scores[best_image_index].item()
        
        print(f"Found best match: {best_image_path} with score {best_score:.4f}")

        # Return the path to the best image and a caption for the UI
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
    description="Enter a text description to search for the most relevant image in the Flickr8k dataset. The app will download the dataset and pre-process images on startup.",
    allow_flagging="never"
)

# This starts the web server
iface.launch()
