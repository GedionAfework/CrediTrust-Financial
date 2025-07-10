import gradio as gr
from src.rag_pipeline import rag_pipeline

def chat_interface(question):
    """Gradio interface function to process user questions."""
    index_path = "vector_store/complaint_index.faiss"
    metadata_path = "vector_store/complaint_metadata.pkl"
    
    answer, retrieved_chunks = rag_pipeline(question, index_path, metadata_path)
    
    sources = "\n\n".join([
        f"**Source {i+1}** (Complaint ID: {chunk['complaint_id']}, Product: {chunk['product']}):\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    
    return answer, sources

with gr.Blocks() as demo:
    gr.Markdown("# CrediTrust Complaint Analysis Chatbot")
    question_input = gr.Textbox(label="Ask a question about customer complaints", placeholder="e.g., Why are people unhappy with BNPL?")
    submit_btn = gr.Button("Submit")
    answer_output = gr.Textbox(label="Answer", interactive=False)
    sources_output = gr.Textbox(label="Retrieved Sources", interactive=False)
    clear_btn = gr.Button("Clear")
    
    submit_btn.click(
        fn=chat_interface,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[question_input, answer_output, sources_output]
    )

if __name__ == "__main__":
    demo.launch()