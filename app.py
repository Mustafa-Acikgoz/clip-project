# app.py (Final Version - Fast Startup)

import gradio as gr
import torch
import pickle
from huggingface_hub import hf_hub_download
from transformers import DistilBertTokenizer

# Import the shared modules from the same repository
import config
from model import CLIPModel

# --- 1. Load ONLY the small, essential files on startup ---
print("Loading application resources...")
device = "cpu"

# Download only the model and data files from your Dataset repo. This is fast.
model_path = hf_hub_download(repo_id=config.REPO_ID, filename=config.SAVED_MODEL_PATH, repo_type="dataset")
embeddings_path = hf_hub_download(repo_id=config.REPO_ID, filename=config.IMAGE_EMBEDDINGS_PATH, repo_type="dataset")
paths_pkl_path = hf_hub_download(repo_id=config.REPO_ID, filename=config.IMAGE_PATHS_PKL, repo_type="dataset")

# Load the model and data
model = CLIPModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

image_embeddings = torch.load(embeddings_path, map_location=device)
with open(paths_pkl_path, 'rb') as f:
    image_filenames = pickle.load(f)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print("✅ Resources loaded successfully.")


# --- 2. Define the search function ---
def search(text_query):
    with torch.no_grad():
        encoded_query = tokenizer([text_query], return_tensors="pt", padding=True, truncation=True)
        text_features = model.text_encoder(
            input_ids=encoded_query['input_ids'].to(device),
            attention_mask=encoded_query['attention_mask'].to(device)
        )
        text_embedding = model.text_projection(text_features)
        
        text_embedding_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        image_embeddings_norm = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        
        similarities = (text_embedding_norm @ image_embeddings_norm.T).squeeze(0)
        best_match_index = similarities.argmax().item()
        
        # --- THE KEY CHANGE IS HERE ---
        # Instead of opening a local file, we construct the public URL to the image
        best_image_filename = image_filenames[best_match_index]
        image_url = f"https://huggingface.co/datasets/{config.REPO_ID}/resolve/main/{config.IMAGE_DIR_NAME}/{best_image_filename}"
        
    return image_url

# --- 3. Create and launch the Gradio Interface ---
iface = gr.Interface(
    fn=search,
    inputs=gr.Textbox(lines=2, placeholder="Enter a description to search for an image..."),
    outputs=gr.Image(type="pil", label="Best Matching Image"),
    title="📷 CLIP-based Image Search",
    description="An image search engine trained on the Flickr8k dataset. Type a description to find the most relevant image.",
    examples=[
        ["a dog running on the beach"],
        ["a child in a pink dress is climbing up a set of stairs"],
        ["a man in a blue shirt is standing on a ladder"]
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()