import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# --- 1. Configuration ---
INPUT_FILE = 'train_data_with_classes.csv'
# Define output file names
CONTENT_EMBEDDINGS_FILE = 'embeddings_content.npy'
COMBINED_EMBEDDINGS_FILE = 'embeddings_combined.npy'

# Use the recommended sentence-transformer model
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# --- 2. Check for GPU and Load Model ---
# Using a GPU will make this process thousands of times faster
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the pre-trained model
# The model will be downloaded from the Hugging Face Hub automatically on first run
print(f"Loading model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, device=device)
print("Model loaded successfully.")


# --- 3. Load Data and Prepare Text for Embedding ---
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Successfully loaded {INPUT_FILE}.")
    
    # Handle any potential missing values in the text columns
    df['content'] = df['content'].fillna('')
    df['inferred company'] = df['inferred company'].fillna('')
    df['username'] = df['username'].fillna('')

    # Create the list of texts to be embedded
    content_texts = df['content'].tolist()
    
    # Create a "combined" text field for richer context
    df['combined_text'] = "Company: " + df['inferred company'] + "; User: " + df['username'] + "; Tweet: " + df['content']
    combined_texts = df['combined_text'].tolist()

except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILE}' was not found. Please run the previous script first.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# --- 4. Generate Embeddings ---
# This is the core step. The model will process the text and output numerical vectors.
print(f"\nGenerating embeddings for {len(content_texts)} content texts...")
content_embeddings = model.encode(content_texts, show_progress_bar=True)
print("Content embeddings generated.")

print(f"\nGenerating embeddings for {len(combined_texts)} combined texts...")
combined_embeddings = model.encode(combined_texts, show_progress_bar=True)
print("Combined embeddings generated.")

# --- 5. Save Embeddings to Files ---
# We save the embeddings as .npy files for efficient storage and loading
np.save(CONTENT_EMBEDDINGS_FILE, content_embeddings)
print(f"\n✅ Content embeddings saved to '{CONTENT_EMBEDDINGS_FILE}'")

np.save(COMBINED_EMBEDDINGS_FILE, combined_embeddings)
print(f"✅ Combined embeddings saved to '{COMBINED_EMBEDDINGS_FILE}'")

print("\nShape of content embeddings:", content_embeddings.shape)
print("Shape of combined embeddings:", combined_embeddings.shape)