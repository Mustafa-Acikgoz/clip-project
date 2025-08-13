# train.py

import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from huggingface_hub import snapshot_download

import config
from dataset import Flickr8kDataset
from model import CLIPModel

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"\rBatch {i+1}/{len(dataloader)} - Loss: {loss.item():.4f}", end='')
        
    return total_loss / len(dataloader)

def main():
    print(f"Downloading dataset from Hugging Face Hub: {config.REPO_ID}")
    dataset_path = snapshot_download(repo_id=config.REPO_ID, repo_type="dataset")
    
    image_dir = os.path.join(dataset_path, config.IMAGE_DIR_NAME)
    caption_file = os.path.join(dataset_path, config.CAPTION_FILE_NAME)
    print("✅ Dataset downloaded.")

    dataset = Flickr8kDataset(image_dir=image_dir, caption_file=caption_file)
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CLIPModel().to(device)
    optimizer = optim.Adam(
        list(model.image_projection.parameters()) + list(model.text_projection.parameters()),
        lr=config.LR
    )

    print("🚀 Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        avg_loss = train_one_epoch(model, data_loader, optimizer, device)
        print(f"\nAverage Epoch Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), config.SAVED_MODEL_PATH)
    print(f"\n🎉 Training complete. Model saved to '{config.SAVED_MODEL_PATH}'")

if __name__ == '__main__':
    main()