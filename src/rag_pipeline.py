import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd
import os

def load_vector_store(index_path, metadata_path):
    """Load FAISS index and metadata with error handling."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found at {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

def embed_question(question, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Embed a user question."""
    model = SentenceTransformer(model_name)
    return model.encode([question])[0]

def retrieve_chunks(question, index, metadata, top_k=5):
    """Retrieve top-k relevant chunks from the vector store."""
    question_embedding = embed_question(question)
    distances, indices = index.search(np.array([question_embedding]), top_k)
    retrieved_chunks = []
    for idx in indices[0]:
        chunk_info = {
            'text': metadata[idx]['text'],
            'complaint_id': metadata[idx]['complaint_id'],
            'product': metadata[idx]['product'],
            'chunk_id': metadata[idx]['chunk_id']
        }
        retrieved_chunks.append(chunk_info)
    return retrieved_chunks

def generate_prompt(question, retrieved_chunks):
    """Generate prompt for the LLM."""
    context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
    prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use only the provided context to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt

def generate_answer(prompt):
    """Generate answer using a lightweight LLM."""
    llm = pipeline("text-generation", model="gpt2", device=-1)
    response = llm(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return response[0]['generated_text'].split("Answer:")[1].strip() if "Answer:" in response[0]['generated_text'] else response[0]['generated_text'].strip()

def rag_pipeline(question, index_path, metadata_path):
    """Run the full RAG pipeline."""
    try:
        index, metadata = load_vector_store(index_path, metadata_path)
        retrieved_chunks = retrieve_chunks(question, index, metadata)
        prompt = generate_prompt(question, retrieved_chunks)
        answer = generate_answer(prompt)
        return answer, retrieved_chunks
    except Exception as e:
        return f"Error in RAG pipeline: {str(e)}", []

def evaluate_rag():
    """Evaluate the RAG pipeline with representative questions."""
    index_path = "vector_store/complaint_index.faiss"
    metadata_path = "vector_store/complaint_metadata.pkl"
    questions = [
        "Why are people unhappy with BNPL?",
        "What are the main issues with Credit card complaints?",
        "Are there any fraud-related complaints for Money transfers?",
        "What problems do customers face with Savings accounts?",
        "How do Personal loan complaints differ from Credit card complaints?"
    ]
    
    evaluation_results = []
    for question in questions:
        answer, retrieved_chunks = rag_pipeline(question, index_path, metadata_path)
        quality_score = 3  # Placeholder; manual scoring needed
        comments = "Generated answer seems relevant but needs validation."
        evaluation_results.append({
            'Question': question,
            'Generated Answer': answer,
            'Retrieved Sources': [f"Complaint ID: {chunk['complaint_id']}, Product: {chunk['product']}, Text: {chunk['text']}" for chunk in retrieved_chunks[:2]],
            'Quality Score': quality_score,
            'Comments': comments
        })
    
    return pd.DataFrame(evaluation_results)

if __name__ == "__main__":
    eval_df = evaluate_rag()
    print(eval_df.to_markdown())
