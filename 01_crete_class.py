import pandas as pd

# Define the input file from the user upload
INPUT_FILE = 'train_data.csv'
OUTPUT_FILE = 'train_data_with_classes.csv'

try:
    # Load the training data
    df = pd.read_csv(INPUT_FILE)
    print(f"Successfully loaded {INPUT_FILE} with {len(df)} rows.")

    # --- 1. Analyze the 'likes' distribution ---
    print("\nAnalyzing distribution of likes:")
    likes_stats = df['likes'].describe()
    print(likes_stats)

    # --- 2. Define class boundaries using quantiles ---
    q75 = likes_stats['75%']
    q95 = df['likes'].quantile(0.95) # Calculate the 95th percentile

    print(f"\nDefining popularity classes based on quantiles:")
    print(f"Class 0 (Common): < {q75:.0f} likes")
    print(f"Class 1 (Popular): {q75:.0f} - {q95:.0f} likes")
    print(f"Class 2 (Viral): > {q95:.0f} likes")

    # --- 3. Create the 'popularity_class' column ---
    def assign_class(likes):
        if likes <= q75:
            return 0
        elif q75 < likes <= q95:
            return 1
        else:
            return 2

    df['popularity_class'] = df['likes'].apply(assign_class)

    # --- 4. Show the distribution of the new classes ---
    print("\nDistribution of new popularity classes:")
    print(df['popularity_class'].value_counts(normalize=True).sort_index())

    # --- 5. Save the new dataframe ---
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Successfully saved data with new 'popularity_class' column to '{OUTPUT_FILE}'")

except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILE}' was not found. Please make sure it's in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")