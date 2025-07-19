import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
from transformers import DistilBertTokenizer

# --- Custom Modules ---
import config
# This is the updated import line to use the new inference-specific model file
from inference_model import CLIPModel 

# --- App Configuration ---
st.set_page_config(page_title="CLIP Image-Text Search", layout="wide")

# --- Model & Tokenizer Loading ---
@st.cache_resource
def load_app_essentials():
    """Load the CLIP model and tokenizer. Cached for performance."""
    device = config.DEVICE
    
    # Instantiate the model with dimensions from config
    model = CLIPModel(
        image_embedding_dim=config.IMAGE_EMBEDDING_DIM,
        text_embedding_dim=config.TEXT_EMBEDDING_DIM,
        projection_dim=config.PROJECTION_DIM
    ).to(device)
    
    # Load the trained model weights
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        model.eval()
    except FileNotFoundError:
        st.error(f"Model file not found at '{config.MODEL_PATH}'. Please ensure it's uploaded to your Hugging Face Space.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model weights: {e}")
        return None, None
        
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return model, tokenizer

# --- Image & Text Processing ---
def preprocess_image(image):
    """Preprocess the image for the model."""
    # This transformation should match the one used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def get_image_from_url(url):
    """Fetch an image from a URL, with error handling."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch image from URL: {e}")
        return None

# --- Main App UI ---
st.title("🖼️ CLIP: Image-Text Search")
st.write("Provide an image and several text descriptions. The app will use your trained CLIP model to find the best textual match for the image.")
st.markdown("---")

# Load the model and tokenizer
model, tokenizer = load_app_essentials()

if model and tokenizer:
    # Create a two-column layout
    col1, col2 = st.columns([0.4, 0.6])

    with col1:
        st.header("Image Input")
        # Let user choose between upload and URL
        source = st.radio("Choose image source:", ["Upload a file", "Enter a URL"], label_visibility="collapsed")
        
        uploaded_image = None
        if source == "Upload a file":
            image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if image_file:
                uploaded_image = Image.open(image_file).convert("RGB")
        else:
            image_url = st.text_input("Image URL:")
            if image_url:
                with st.spinner("Fetching image..."):
                    uploaded_image = get_image_from_url(image_url)

        if uploaded_image:
            st.image(uploaded_image, caption="Your Image", use_column_width=True)

    with col2:
        st.header("Text Queries")
        st.write("Enter one text description per line.")
        # Use a text area for multiple queries
        text_queries_input = st.text_area(
            "Text descriptions", 
            "a dog running on the beach\na person standing in front of a building\na cat sleeping on a chair\nmountains under a blue sky",
            height=200,
            label_visibility="collapsed"
        )
        # Process input into a clean list of queries
        queries = [q.strip() for q in text_queries_input.split('\n') if q.strip()]

    st.markdown("---")

    # The main action button
    if st.button("🔍 Find Best Match", use_container_width=True) and uploaded_image and queries:
        with st.spinner("Analyzing image and text..."):
            # Process the image to get a tensor
            image_tensor = preprocess_image(uploaded_image).to(config.DEVICE)
            
            # Process the text queries to get tokens
            text_inputs = tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(config.DEVICE)
            
            # Get model embeddings using the modified forward pass
            with torch.no_grad():
                image_embedding, text_embeddings = model(
                    image_features=image_tensor,
                    text_input_ids=text_inputs['input_ids'], 
                    text_attention_mask=text_inputs['attention_mask']
                )

            # Calculate cosine similarity
            image_embedding_norm = F.normalize(image_embedding, p=2, dim=-1)
            text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=-1)
            similarity_scores = (image_embedding_norm @ text_embeddings_norm.T).squeeze(0)
            
            # Convert scores to probabilities using softmax for a more intuitive result
            similarity_probs = F.softmax(similarity_scores, dim=-1)

            st.header("🏆 Results")
            # Create a sorted list of (query, score) tuples
            results = sorted(zip(queries, similarity_probs.tolist()), key=lambda x: x[1], reverse=True)
            
            # Display results with progress bars
            for query, score in results:
                st.write(f'**{score:.2%} match:** "{query}"')
                st.progress(score)

elif not model or not tokenizer:
    st.warning("Application could not start. Please check the logs on your Hugging Face Space for any errors during model loading.")

