# inference_model.py
import torch
import torch.nn as nn
from torchvision.models import resnet50
from transformers import DistilBertModel

# --- Copy these classes from your original file ---
class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Note: Using the newer 'weights' parameter is recommended
        pretrained_resnet50 = resnet50(weights='IMAGENET1K_V1')
        self.model = nn.Sequential(*list(pretrained_resnet50.children())[:-1])
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), -1)

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=256, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

# --- This is the MODIFIED CLIPModel for inference ---
class CLIPModel(nn.Module):
    def __init__(self, image_embedding_dim, text_embedding_dim, projection_dim):
        super().__init__()
        self.vision_encoder = VisionEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding_dim, projection_dim=projection_dim)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_dim, projection_dim=projection_dim)

    def forward(self, image_features=None, text_input_ids=None, text_attention_mask=None):
        image_embedding = None
        if image_features is not None:
            image_features = self.vision_encoder(image_features)
            image_embedding = self.image_projection(image_features)

        text_embedding = None
        if text_input_ids is not None:
            text_features = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
            text_embedding = self.text_projection(text_features)

        return image_embedding, text_embedding