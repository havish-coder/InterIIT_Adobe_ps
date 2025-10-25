import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib

# --- 1. Configuration ---
DATA_FILE = 'train_data_with_classes.csv'
EMBEDDINGS_FILE = 'embeddings_content.npy'
MODEL_OUTPUT_FILE = 'classifier_model_improved.joblib'

# --- 2. Load Data and Embeddings ---
try:
    df = pd.read_csv(DATA_FILE)
    y = df['popularity_class'].values
    print(f"Successfully loaded labels from {DATA_FILE}.")

    X = np.load(EMBEDDINGS_FILE)
    print(f"Successfully loaded embeddings from {EMBEDDINGS_FILE}.")

    if len(X) != len(y):
        raise ValueError("Mismatch between number of embeddings and labels.")

except FileNotFoundError:
    print(f"Error: Make sure '{DATA_FILE}' and '{EMBEDDINGS_FILE}' are present.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# --- 3. Split Data into Training and Validation Sets ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nData split into training ({len(X_train)} samples) and validation ({len(X_val)} samples).")


# --- 4. NEW: Calculate Class Weights to Counter Imbalance ---
print("\nCalculating class weights to add bias for rare classes...")
# This function calculates the weights needed to balance the dataset.
# 'balanced' mode automatically adjusts weights inversely proportional to class frequencies.
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

print(f"Calculated weights: {class_weights}")
# For XGBoost, we need to create a sample_weight array for the training data
sample_weights = np.array([class_weights[c] for c in y_train])


# --- 5. Train the XGBoost Classifier with Weights ---
print("\nTraining the XGBoost classifier with bias (class weights)...")

classifier = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    n_estimators=500, # Increased estimators for more learning capacity
    learning_rate=0.05,
    max_depth=8, # Increased depth for more complex patterns
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# ** KEY CHANGE: Pass the 'sample_weight' parameter during training **
classifier.fit(X_train, y_train,
             sample_weight=sample_weights, # <--- THIS IS THE FIX
             eval_set=[(X_val, y_val)],
             early_stopping_rounds=40,
             verbose=False)

print("Classifier training complete.")


# --- 6. Evaluate the Improved Classifier ---
print("\nEvaluating classifier performance on the validation set...")
y_pred = classifier.predict(X_val)

print("\nClassification Report (Improved):")
print(classification_report(y_val, y_pred, target_names=['Class 0 (Common)', 'Class 1 (Popular)', 'Class 2 (Viral)']))

print("\nConfusion Matrix (Improved):")
print(confusion_matrix(y_val, y_pred))


# --- 7. Save the Trained Model ---
joblib.dump(classifier, MODEL_OUTPUT_FILE)
print(f"\n✅ Successfully trained and saved the IMPROVED classifier model to '{MODEL_OUTPUT_FILE}'")