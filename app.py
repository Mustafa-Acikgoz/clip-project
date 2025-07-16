# app.py
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DistilBertTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader

import config
from model import CLIPModel
from dataset import Flickr8kDataset

st.set_page_config(layout="wide")
st.title("🖼️ CLIP-Style Image Search Engine (Textbook Version)")
st.write("This app uses a model trained according to the textbook's architecture to find images based on text descriptions. The encoders are frozen, and only the projection heads were trained.")

@st.cache_resource
def load_model_and_tokenizer():
    """Loads the trained CLIP model and tokenizer once."""
    model = CLIPModel(
        image_embedding_dim=config.IMAGE_EMBEDDING_DIM,
        text_embedding_dim=config.TEXT_EMBEDDING_DIM,
        projection_dim=config.PROJECTION_DIM
    ).to(config.DEVICE)
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    except FileNotFoundError:
        st.error(f"Model file not found at '{config.MODEL_PATH}'. Please train the model first by running `python train.py`.")
        return None, None
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return model, tokenizer

@st.cache_data
def get_all_image_embeddings(_model):
    """Computes and caches embeddings for all images in the dataset."""
    dataset = Flickr8kDataset(config.IMAGE_DIR, config.CAPTION_FILE, tokenizer=None)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Pre-computing image embeddings"):
            images = batch['image'].to(config.DEVICE)
            image_features = _model.vision_encoder(images)
            image_embeddings = _model.image_projection(image_features)
            all_embeddings.append(image_embeddings)
    
    return torch.cat(all_embeddings), dataset.image_paths

model, tokenizer = load_model_and_tokenizer()

if model and tokenizer:
    image_embeddings, image_paths = get_all_image_embeddings(model)
    st.success("Model loaded and image embeddings are ready!")

    st.header("Search for an Image")
    query = st.text_input("Enter your search query:", "a dog running on a beach")

    if st.button("Search"):
        if not query:
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching..."):
                encoded_query = tokenizer([query], return_tensors='pt', padding=True, truncation=True)
                batch = {k: v.to(config.DEVICE) for k, v in encoded_query.items()}
                
                with torch.no_grad():
                    text_features = model.text_encoder(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )
                    text_embedding = model.text_projection(text_features)
                
                text_embedding_norm = F.normalize(text_embedding, p=2, dim=-1)
                image_embeddings_norm = F.normalize(image_embeddings, p=2, dim=-1)
                
                dot_similarity = text_embedding_norm @ image_embeddings_norm.T
                values, indices = torch.topk(dot_similarity.squeeze(0), 9)
                top_image_paths = [image_paths[i] for i in indices.cpu().numpy()]

            st.success(f"Top {len(top_image_paths)} results for '{query}':")
            cols = st.columns(3)
            for i, path in enumerate(top_image_paths):
                try:
                    cols[i % 3].image(Image.open(path), use_column_width=True, caption=f"Result {i+1}")
                except Exception as e:
                    cols[i % 3].error(f"Could not load image {i+1}.\nError: {e}")