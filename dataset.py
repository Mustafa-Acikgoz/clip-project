# dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_file):
        self.image_dir = image_dir
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.image_paths, self.captions = self._read_caption_file(caption_file)
        
        self.caption_encodings = self.tokenizer(
            self.captions, 
            truncation=True, 
            padding=True, 
            max_length=200,
            return_tensors="pt"
        )

    def _read_caption_file(self, caption_file):
        image_paths = []
        captions = []
        with open(caption_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines[1:]: # Skip header
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    image_name, caption = parts
                    image_paths.append(os.path.join(self.image_dir, image_name))
                    captions.append(caption)
        return image_paths, captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.caption_encodings.items()}
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        item['image'] = img
        return item