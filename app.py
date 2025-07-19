import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
from transformers import DistilBertTokenizer
import config # Your config file
from inference_model import CLIPModel # Your model class file

# --- 1. Load Model and Tokenizer (runs only once) ---
# This section loads your trained model and tokenizer when the app starts.
device = config.DEVICE

# Load model with dimensions from config
model = CLIPModel(
    image_embedding_dim=config.IMAGE_EMBEDDING_DIM,
    text_embedding_dim=config.TEXT_EMBEDDING_DIM,
    projection_dim=config.PROJECTION_DIM
).to(device)

# Load the trained model weights from your .pth file
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
model.eval()

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print("Model and Tokenizer loaded successfully.")

# --- 2. Image Preprocessing Function (reused from your code) ---
def preprocess_image(image):
    """Preprocess the image for the model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# --- 3. The Main Gradio Function ---
# This is the core function that Gradio will build a UI around.
# It takes the inputs from the UI and returns the outputs to the UI.
def find_best_match(image_input, text_queries_input):
    """
    Takes an image and a block of text queries, and returns a dictionary
    of queries and their similarity scores.
    """
    if image_input is None:
        return "Please provide an image."
    if not text_queries_input:
        return "Please provide text descriptions."

    # Process the image to get a tensor
    image_tensor = preprocess_image(image_input).to(device)

    # Process the text queries into a clean list
    queries = [q.strip() for q in text_queries_input.split('\n') if q.strip()]
    if not queries:
        return "Please provide valid text descriptions."

    # Process the text queries to get tokens
    text_inputs = tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(device)

    # Get model embeddings
    with torch.no_grad():
        image_embedding, text_embeddings = model(
            image_features=image_tensor,
            text_input_ids=text_inputs['input_ids'],
            text_attention_mask=text_inputs['attention_mask']
        )

    # Calculate cosine similarity and format for Gradio's Label component
    image_embedding_norm = F.normalize(image_embedding, p=2, dim=-1)
    text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=-1)
    similarity_scores = (image_embedding_norm @ text_embeddings_norm.T).squeeze(0)
    
    # Create a results dictionary: { "query text": score, ... }
    results = {query: score.item() for query, score in zip(queries, similarity_scores)}
    
    return results

# --- 4. Create and Launch the Gradio Interface ---
iface = gr.Interface(
    fn=find_best_match,
    inputs=[
        gr.Image(type="pil", label="Upload or Drag an Image"),
        gr.Textbox(lines=5, label="Text Descriptions (one per line)", placeholder="a person on a beach\na black cat\na city skyline at night")
    ],
    outputs=gr.Label(num_top_classes=5, label="Results"),
    title="🖼️ CLIP Image-Text Search",
    description="Provide an image and several text descriptions. The app will use a trained CLIP model to find the best textual match for the image."
)

iface.launch()