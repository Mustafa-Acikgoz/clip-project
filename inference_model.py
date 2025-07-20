import torch
import torch.nn as nn
from torchvision.models import resnet50
from transformers import DistilBertModel

# --- Helper Classes (VisionEncoder, TextEncoder, ProjectionHead) ---
# These define the components of the overall CLIP model.

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the recommended 'weights' parameter for pre-trained models
        pretrained_resnet50 = resnet50(weights='IMAGENET1K_V1')
        # Use all layers of ResNet50 except for the final fully connected layer
        self.model = nn.Sequential(*list(pretrained_resnet50.children())[:-1])
        # Freeze the parameters of the vision encoder
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        # Flatten the output to a 1D tensor per image
        return x.view(x.size(0), -1)

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Freeze the parameters of the text encoder
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the embedding of the [CLS] token as the sentence representation
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
        # Add a residual connection
        x = x + projected
        x = self.layer_norm(x)
        return x

# --- Main CLIPModel for Inference ---
# This class combines the encoders and projection heads.

class CLIPModel(nn.Module):
    def __init__(self, image_embedding_dim, text_embedding_dim, projection_dim):
        super().__init__()
        
        self.image_encoder = VisionEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding_dim, projection_dim=projection_dim)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding_dim, projection_dim=projection_dim)

    def forward(self, image_features=None, text_input_ids=None, attention_mask=None):
        """
        This forward pass handles both image and text inputs.
        app.py will call this to get the final, projected embeddings.
        
        **MODIFICATION**: Renamed 'text_attention_mask' to 'attention_mask' for
        compatibility with the standard Hugging Face tokenizer output.
        """
        image_embedding = None
        if image_features is not None:
            # Get raw features from the vision backbone
            image_features_raw = self.image_encoder(image_features)
            # Project them into the shared embedding space
            image_embedding = self.image_projection(image_features_raw)

        text_embedding = None
        if text_input_ids is not None:
            # Get raw features from the text backbone
            text_features_raw = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=attention_mask
            )
            # Project them into the shared embedding space
            text_embedding = self.text_projection(text_features_raw)

        return image_embedding, text_embedding