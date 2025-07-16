# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms

class Flickr8kDataset(Dataset):
    """
    Custom PyTorch Dataset for the Flickr8k data.
    It loads images and their corresponding captions, tokenizing the text
    on initialization for efficiency.
    """
    def __init__(self, image_dir, caption_file, tokenizer):
        self.image_dir = image_dir
        self.tokenizer = tokenizer

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        df = pd.read_csv(caption_file)
        self.image_paths = [os.path.join(self.image_dir, fname) for fname in df['image']]
        self.captions = df['caption'].tolist()

        print("Tokenizing all captions... (This may take a moment)")
        self.caption_encodings = self.tokenizer(
            self.captions,
            truncation=True,
            padding='max_length',
            max_length=200,
            return_tensors="pt"
        )
        print("Tokenization complete.")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.caption_encodings.items()}
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            item['image'] = self.transform(img)
        except (FileNotFoundError):
            print(f"Warning: Could not load image at {self.image_paths[idx]}. Returning a black image.")
            item['image'] = torch.zeros((3, 224, 224))
        item["caption_text"] = self.captions[idx]
        return item