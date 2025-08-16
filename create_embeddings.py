# create_embeddings.py (Corrected Version)

import torch
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download
import os
import pickle

import config
from model import CLIPModel
from dataset import Flickr8kDataset

def create_embedding_index():
    print(f"Downloading dataset and model from Hugging Face Hub: {config.REPO_ID}")
    repo_path = snapshot_download(repo_id=config.REPO_ID, repo_type="dataset")
    
    image_dir = os.path.join(repo_path, config.IMAGE_DIR_NAME)
    caption_file = os.path.join(repo_path, config.CAPTION_FILE_NAME)
    model_path = os.path.join(repo_path, config.SAVED_MODEL_PATH)
    print("✅ Repository downloaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading trained CLIP model...")
    model = CLIPModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded.")

    dataset = Flickr8kDataset(image_dir=image_dir, caption_file=caption_file)
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE)
    
    # --- CHANGE IS HERE ---
    # Get only the filename of each image, not the full path
    all_image_filenames = [os.path.basename(p) for p in dataset.image_paths]

    all_image_embeddings = []
    print("🚀 Generating image embeddings...")
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            image_features = model.vision_encoder(batch['image'].to(device))
            image_embeddings = model.image_projection(image_features)
            all_image_embeddings.append(image_embeddings.cpu())
            print(f"\rProcessed batch {i+1}/{len(data_loader)}", end='')

    full_embeddings_tensor = torch.cat(all_image_embeddings)
    torch.save(full_embeddings_tensor, 'image_embeddings.pt')
    
    # --- CHANGE IS HERE ---
    # Save the list of filenames
    with open('image_paths.pkl', 'wb') as f:
        pickle.dump(all_image_filenames, f)

    print("\n\n🎉 Index creation complete.")
    print(f"Embeddings saved to 'image_embeddings.pt'")
    print(f"Image filenames saved to 'image_paths.pkl'")

if __name__ == '__main__':
    create_embedding_index()