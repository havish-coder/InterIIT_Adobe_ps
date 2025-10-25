import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import joblib
import torch

# --- 1. Configuration ---
# Load the models we trained in the previous steps
CLASSIFIER_MODEL_FILE = 'classifier_model_improved.joblib'
REGRESSOR_CLASS_0_FILE = 'regressor_model_class_0.joblib'
REGRESSOR_CLASS_1_FILE = 'regressor_model_class_1.joblib'
REGRESSOR_CLASS_2_FILE = 'regressor_model_class_2.joblib'

# The sentence transformer model for generating embeddings
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# The original data file
DATA_FILE = 'train_data.csv' 
OUTPUT_FILE = 'final_predictions.csv'


# --- 2. Load Models and Embedding Generator ---
try:
    print("Loading all trained models...")
    classifier = joblib.load(CLASSIFIER_MODEL_FILE)
    regressor_c0 = joblib.load(REGRESSOR_CLASS_0_FILE)
    regressor_c1 = joblib.load(REGRESSOR_CLASS_1_FILE)
    regressor_c2 = joblib.load(REGRESSOR_CLASS_2_FILE)
    print("All models loaded successfully.")

    print(f"Loading sentence transformer model: {EMBEDDING_MODEL_NAME}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    print("Embedding model loaded successfully.")

except FileNotFoundError as e:
    print(f"Error: Model file not found - {e}. Please run the training scripts first.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()


# --- 3. Prepare Test Data ---
# For this demonstration, we'll use the last 20% of our original data as a "test set"
try:
    df_full = pd.read_csv(DATA_FILE)
    # Simple split for demonstration purposes
    split_point = int(len(df_full) * 0.8)
    df_test = df_full.iloc[split_point:].copy()
    print(f"\nCreated a test set with {len(df_test)} samples.")
    
    # Handle missing values
    df_test['content'] = df_test['content'].fillna('')
    df_test['inferred company'] = df_test['inferred company'].fillna('')
    df_test['username'] = df_test['username'].fillna('')

except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found.")
    exit()


# --- 4. Generate Embeddings for the Test Data ---
print("Generating embeddings for the test data...")

# Create the two types of embeddings needed
content_texts = df_test['content'].tolist()
df_test['combined_text'] = "Company: " + df_test['inferred company'] + "; User: " + df_test['username'] + "; Tweet: " + df_test['content']
combined_texts = df_test['combined_text'].tolist()

content_embeddings = embedding_model.encode(content_texts, show_progress_bar=True)
combined_embeddings = embedding_model.encode(combined_texts, show_progress_bar=True)

print("Embeddings generated.")


# --- 5. Run the Full Prediction Pipeline ---
print("\nStarting the prediction pipeline...")

# a. Predict the popularity CLASS for each tweet using the classifier
# The classifier was trained on content embeddings
predicted_classes = classifier.predict(content_embeddings)
df_test['predicted_class'] = predicted_classes
print("Step 1: Classification complete.")

# b. Predict the LIKES using the appropriate specialist regressor
predictions = []
for index, row in df_test.iterrows():
    predicted_class = row['predicted_class']
    # We need the combined embedding for the regressors
    embedding_vector = combined_embeddings[df_test.index.get_loc(index)].reshape(1, -1)
    
    prediction_log = 0
    if predicted_class == 0:
        prediction_log = regressor_c0.predict(embedding_vector)
    elif predicted_class == 1:
        prediction_log = regressor_c1.predict(embedding_vector)
    else: # Class is 2
        prediction_log = regressor_c2.predict(embedding_vector)
        
    predictions.append(prediction_log[0])

# Convert log predictions back to the original scale
final_predictions = np.expm1(predictions)
final_predictions[final_predictions < 0] = 0 # Ensure no negative likes
df_test['predicted_likes'] = final_predictions.astype(int)
print("Step 2: Regression complete.")


# --- 6. Save the Final Output ---
# Select and reorder columns for a clean output file
output_df = df_test[['content', 'likes', 'predicted_class', 'predicted_likes']].copy()
output_df.rename(columns={'likes': 'actual_likes'}, inplace=True)

output_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Pipeline complete! Final predictions saved to '{OUTPUT_FILE}'.")
print("\nHere's a preview of your results:")
print(output_df.head(10))