# CLIP-Style Image Search Engine (Textbook Implementation)

This project provides a complete, modular, and end-to-end implementation of a CLIP-style model for text-to-image search. The architecture and training methodology are a faithful reproduction of the approach described in Chapter 14 of the textbook, "Building an Image Search Engine Using CLIP: a Multimodal Approach".

The project is structured for clarity and maintainability, making it an ideal portfolio piece to showcase skills in PyTorch, model implementation, and MLOps practices like deployment with Streamlit and Hugging Face.

## Key Features

- **Faithful "Book Version" Architecture:** Implements the specific design choices from the textbook:
  - **Frozen Vision Encoder:** Uses a pre-trained `ResNet50` as a fixed feature extractor.
  - **Frozen Text Encoder:** Uses a pre-trained `DistilBERT` as a fixed feature extractor.
  - **Projection Heads:** Maps both image and text features into a shared 256-dimensional space.
  - **Custom Contrastive Loss:** Implements the unique loss function described in the book.
- **Modular & Professional Code Structure:** The code is separated into logical files (`config.py`, `dataset.py`, `model.py`, `train.py`, `app.py`) for better organization and scalability.
- **End-to-End MLOps Pipeline:**
  - **Training:** A dedicated script to train the model and save the weights.
  - **Inference:** A standalone Streamlit web application for interactive text-to-image search.
  - **Hub Integration:** Detailed instructions for uploading the trained model and hosting the app on the Hugging Face Hub.

## Project Structure
your-clip-project/
│
├── data/
│   ├── images/
│   └── captions.txt
│
├── app.py
├── config.py
├── dataset.py
├── model.py
├── train.py
│
├── requirements.txt
└── README.md


## Setup and Installation

**1. Clone the Repository:**
```bash
git clone <your-repo-url>
cd your-clip-project
2. Create a Python Virtual Environment:

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install Dependencies:

Bash

pip install -r requirements.txt
4. Download the Flickr8k Dataset:

Request the dataset from the official source: https://illinois.edu/fb/sec/1713398.

Download and extract Flickr8k_Dataset.zip into the data/images/ folder.

Find a captions.txt file (commonly available on Kaggle versions of the dataset) and place it at data/captions.txt.

How to Run
Step 1: Train the Model
First, you must train the model. This will create a clip_book_model.pth file containing the learned weights of the projection heads.

Run the training script from your terminal:

Bash

python train.py
Step 2: Launch the Web Application
Once the model is trained, launch the interactive search engine with Streamlit:

Bash

streamlit run app.py
This will open a new tab in your browser with the application running.