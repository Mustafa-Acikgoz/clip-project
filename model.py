# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import DistilBertModel
import config

class VisionEncoder(nn.Module):
    """Encodes images into a fixed-size vector using a pretrained ResNet50."""
    def __init__(self):
        super().__init__()
        # Use updated `weights` argument
        pretrained_resnet50 = resnet50(weights='IMAGENET1K_V1')
        self.model = nn.Sequential(*list(pretrained_resnet50.children())[:-1])
        # Freeze parameters as we are using it as a feature extractor
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), -1)

class TextEncoder(nn.Module):
    """Encodes text into a fixed-size vector using a pretrained DistilBERT."""
    def __init__(self):
        super().__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token's embedding as the sentence representation
        return outputs.last_hidden_state[:, 0, :]

class ProjectionHead(nn.Module):
    """Projects embeddings into a shared, lower-dimensional space."""
    def __init__(self, embedding_dim, projection_dim=config.PROJECTION_DIM):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(projection_dim)
        
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected # Add residual connection
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    """The main model that combines vision and text encoders and calculates loss."""
    def __init__(self):
        super().__init__()
        self.vision_encoder = VisionEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=config.IMAGE_EMBEDDING_DIM)
        self.text_projection = ProjectionHead(embedding_dim=config.TEXT_EMBEDDING_DIM)

    def forward(self, batch):
        # Get embeddings
        image_features = self.vision_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"]
        )
        
        # Project into the shared space
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        
        # Calculate contrastive loss
        logits = text_embeddings @ image_embeddings.T
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2, dim=-1
        )
        texts_loss = F.cross_entropy(logits, targets)
        images_loss = F.cross_entropy(logits.T, targets.T)
        
        loss = (images_loss + texts_loss) / 2.0
        return loss