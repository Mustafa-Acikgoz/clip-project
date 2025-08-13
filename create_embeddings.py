import torch
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download
import os
import pickle

# Import necessary classes from your shared project files
import config
from model import CLIPModel
from dataset import Flickr8kDataset

def create_embedding_index():
    """
    Generates and saves embeddings for all images in the dataset.
    This is a one-time inference process that prepares data for the app.
    """
    # --- 1. Download the repository which contains the model and data ---
    print(f"Downloading dataset and model from Hugging Face Hub: {config.REPO_ID}")
    # This command downloads the entire repository content to a local cache
    repo_path = snapshot_download(repo_id=config.REPO_ID, repo_type="dataset")
    
    # Define paths to the downloaded files within the cache
    image_dir = os.path.join(repo_path, config.IMAGE_DIR_NAME)
    caption_file = os.path.join(repo_path, config.CAPTION_FILE_NAME)
    model_path = os.path.join(repo_path, config.SAVED_MODEL_PATH)
    print("✅ Repository downloaded.")

    # --- 2. Load the trained model for inference ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading trained CLIP model...")
    model = CLIPModel().to(device)
    # Load the state dictionary from the downloaded .pth file
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set the model to evaluation/inference mode
    print("✅ Model loaded.")

    # --- 3. Prepare the dataset ---
    dataset = Flickr8kDataset(image_dir=image_dir, caption_file=caption_file)
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE)
    # Get all original image paths for saving later
    all_image_paths = dataset.image_paths 

    # --- 4. Generate and collect embeddings ---
    all_image_embeddings = []
    print("🚀 Generating image embeddings...")
    with torch.no_grad(): # Disable gradient calculations for inference
        for i, batch in enumerate(data_loader):
            # Use only the vision part of the model
            image_features = model.vision_encoder(batch['image'].to(device))
            image_embeddings = model.image_projection(image_features)
            # Move embeddings to CPU before appending to avoid GPU memory buildup
            all_image_embeddings.append(image_embeddings.cpu())
            print(f"\rProcessed batch {i+1}/{len(data_loader)}", end='')

    # --- 5. Save the results to your local project folder ---
    # Concatenate all batch embeddings into a single tensor
    full_embeddings_tensor = torch.cat(all_image_embeddings)
    
    # Save the embeddings tensor
    torch.save(full_embeddings_tensor, 'image_embeddings.pt')
    
    # Save the list of image paths using pickle
    with open('image_paths.pkl', 'wb') as f:
        pickle.dump(all_image_paths, f)

    print("\n\n🎉 Index creation complete.")
    print(f"Embeddings saved to 'image_embeddings.pt'")
    print(f"Image paths saved to 'image_paths.pkl'")

if __name__ == '__main__':
    create_embedding_index()