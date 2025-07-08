import pandas as pd
import re
import os

def load_data(file_path):
    """Load the CFPB complaint dataset."""
    return pd.read_csv(file_path)

def clean_text(text):
    """Clean text by lowercasing and removing special characters."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def preprocess_data(df, target_products, output_path):
    """Filter and clean the dataset for specified products."""
    df_filtered = df[df['Product'].isin(target_products)].copy()
    
    df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].notnull()]
    
    df_filtered['Consumer complaint narrative'] = df_filtered['Consumer complaint narrative'].apply(clean_text)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_filtered.to_csv(output_path, index=False)
    return df_filtered

if __name__ == "__main__":
    input_path = "data/raw/complaints.csv"
    output_path = "data/filtered_complaints.csv"
    target_products = ['Credit card', 'Consumer Loan', 'Payday loan', 'Checking or savings account', 'Money transfer']
    
    df = load_data(input_path)
    df_filtered = preprocess_data(df, target_products, output_path)
    print(f"Filtered dataset saved to {output_path}, shape: {df_filtered.shape}")