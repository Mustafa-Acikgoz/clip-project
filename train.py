# train.py
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from tqdm import tqdm

import config
from dataset import Flickr8kDataset
from model import CLIPModel

def main():
    print(f"--- Starting Training ---")
    print(f"Using device: {config.DEVICE}")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = Flickr8kDataset(config.IMAGE_DIR, config.CAPTION_FILE, tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = CLIPModel(
        image_embedding_dim=config.IMAGE_EMBEDDING_DIM,
        text_embedding_dim=config.TEXT_EMBEDDING_DIM,
        projection_dim=config.PROJECTION_DIM
    ).to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")

        for batch in progress_bar:
            batch = {k: v.to(config.DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            loss = model(batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

    print("\nTraining complete.")
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")
    print("\nTo upload to Hugging Face Hub, run the upload_to_hub.py script.")

if __name__ == '__main__':
    main()