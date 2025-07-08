import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def chunk_text(text, chunk_size=200, chunk_overlap=20):
    """Chunk text using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

def generate_embeddings(texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Generate embeddings using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True)

def create_vector_store(embeddings, metadata, index_path, metadata_path):
    """Create and save FAISS vector store with metadata."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

def process_complaints(input_path, vector_store_path, metadata_path):
    """Process complaints: chunk, embed, and index."""
    df = pd.read_csv(input_path)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = []
    metadata = []
    
    for idx, row in df.iterrows():
        complaint_chunks = chunk_text(row['Consumer complaint narrative'])
        for i, chunk in enumerate(complaint_chunks):
            chunks.append(chunk)
            metadata.append({
                'complaint_id': row['Complaint ID'],
                'product': row['Product'],
                'chunk_id': f"{row['Complaint ID']}_{i}"
            })
    
    embeddings = generate_embeddings(chunks)
    
    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
    create_vector_store(embeddings, metadata, vector_store_path, metadata_path)
    
    print(f"Vector store saved to {vector_store_path}")
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    input_path = "data/filtered_complaints.csv"
    vector_store_path = "vector_store/complaint_index.faiss"
    metadata_path = "vector_store/complaint_metadata.pkl"
    
    process_complaints(input_path, vector_store_path, metadata_path)