import pandas as pd
import re
import os

def load_data(file_path):
    """Load the CFPB complaint dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with {len(df)} rows")
    print("Unique products:", df['Product'].unique().tolist())
    return df

def clean_text(text):
    """Clean text by lowercasing and removing special characters."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def preprocess_data(df, target_products, output_path):
    """Filter and clean the dataset for specified products."""
    print(f"Target products: {target_products}")
    
    # Normalize product names for comparison
    df['Product'] = df['Product'].str.strip()
    df_filtered = df[df['Product'].isin(target_products)].copy()
    print(f"Rows after product filtering: {len(df_filtered)}")
    
    # Filter out null or empty narratives
    df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].notnull() & (df_filtered['Consumer complaint narrative'].str.strip() != '')]
    print(f"Rows after narrative filtering: {len(df_filtered)}")
    
    if df_filtered.empty:
        raise ValueError("No rows remain after filtering. Check product names or narrative data.")
    
    df_filtered['Consumer complaint narrative'] = df_filtered['Consumer complaint narrative'].apply(clean_text)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_filtered.to_csv(output_path, index=False)
    return df_filtered

if __name__ == "__main__":
    input_path = "data/raw/complaints.csv"
    output_path = "data/filtered_complaints.csv"
    target_products = ['Credit card', 'Consumer Loan', 'Payday loan', 'Checking or savings account', 'Money transfer']
    
    try:
        df = load_data(input_path)
        df_filtered = preprocess_data(df, target_products, output_path)
        print(f"Filtered dataset saved to {output_path}, shape: {df_filtered.shape}")
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")