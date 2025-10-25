import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# --- 1. Configuration ---
DATA_FILE = 'train_data_with_classes.csv'
# For regression, we'll use the richer 'combined' embeddings
EMBEDDINGS_FILE = 'embeddings_combined.npy' 

# --- 2. Load Data and Embeddings ---
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Successfully loaded data from {DATA_FILE}.")

    # Load the pre-computed embeddings
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"Successfully loaded embeddings from {EMBEDDINGS_FILE}.")

    if len(df) != len(embeddings):
        raise ValueError("Mismatch between number of data points and embeddings.")

except FileNotFoundError:
    print(f"Error: Make sure '{DATA_FILE}' and '{EMBEDDINGS_FILE}' are in the same directory.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()


# --- 3. Loop Through Each Class to Train a Specialist Regressor ---
# We will train one model for each class: 0, 1, and 2.
for class_label in sorted(df['popularity_class'].unique()):
    print(f"\n--- Training Regressor for Class {class_label} ---")

    # a. Filter the data to get only the tweets for the current class
    class_df = df[df['popularity_class'] == class_label]
    class_indices = class_df.index
    
    X_class = embeddings[class_indices]
    # We predict the log of likes for better stability, then convert back later
    y_class_log = np.log1p(class_df['likes'].values)

    print(f"Found {len(X_class)} samples for this class.")

    # b. Split the class-specific data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_class, y_class_log,
        test_size=0.2,
        random_state=42
    )

    # c. Initialize and train the XGBoost Regressor
    # Parameters can be tuned for each class, but we'll start with robust defaults
    regressor = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        early_stopping_rounds=50,
        n_jobs=-1  # Use all available CPU cores
    )

    print("Training model...")
    regressor.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

    # d. Evaluate the specialist model on its validation set
    preds_log = regressor.predict(X_val)
    preds = np.expm1(preds_log) # Convert predictions back to original scale
    y_val_original = np.expm1(y_val)
    
    rmse = np.sqrt(mean_squared_error(y_val_original, preds))
    print(f"Validation RMSE for Class {class_label} model: {rmse:.2f}")

    # e. Save the trained specialist model to a file
    model_output_file = f'regressor_model_class_{class_label}.joblib'
    joblib.dump(regressor, model_output_file)
    print(f"✅ Model for Class {class_label} saved to '{model_output_file}'")

print("\n--- All specialist models have been trained and saved! ---")