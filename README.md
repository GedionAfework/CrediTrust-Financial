# CrediTrust Financial Complaint Analysis

## Project Overview
This project develops an AI-powered complaint analysis chatbot for CrediTrust Financial, a digital finance company in East Africa. The system leverages a Retrieval-Augmented Generation (RAG) pipeline to process unstructured customer complaints from the Consumer Financial Protection Bureau (CFPB) dataset, enabling stakeholders to query trends in plain English. The project addresses Key Performance Indicators (KPIs) by reducing analysis time from days to minutes, empowering non-technical users, and enabling proactive issue identification. The system focuses on five product categories: Credit Card, Consumer Loan, Payday Loan, Checking or Savings Account, and Money Transfer.

The project was executed in four tasks:
1. **Exploratory Data Analysis (EDA) and Preprocessing**: Analyze and clean the CFPB dataset.
2. **Text Chunking and Embedding**: Convert narratives into embeddings and index them in a FAISS vector store.
3. **RAG Pipeline and Evaluation**: Build and evaluate a pipeline for retrieving and generating answers.
4. **Interactive Chat Interface**: Develop a Gradio-based web interface for querying.


## Project Structure
```
creditrust-complaint-analysis/
├── .venv/                             # Virtual environment
├── data/
│   ├── raw/
│   │   └── complaints.csv            # Raw CFPB dataset
│   └── filtered_complaints.csv       # Filtered dataset
├── docs/
│   ├── evaluation_table.md           # RAG evaluation results
│   └── report.pdf                    # Technical report
├── notebooks/
│   └── eda_preprocessing.ipynb       # EDA and preprocessing notebook
├── src/
│   ├── preprocessing.py              # Preprocessing script
│   ├── embedding.py                  # Chunking and embedding script
│   ├── rag_pipeline.py               # RAG pipeline implementation
│   └── app.py                        # Gradio interface
├── vector_store/
│   ├── complaint_index.faiss         # FAISS vector store
│   └── complaint_metadata.pkl        # Metadata for embeddings
├── requirements.txt                  # Project dependencies
└── README.md                         # This file
```

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd creditrust-complaint-analysis
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download CFPB Dataset**:
   - Place the CFPB complaints dataset in `data/raw/complaints.csv`. Download from the [CFPB website](https://www.consumerfinance.gov/data-research/consumer-complaints/) if needed.

5. **Run the Pipeline**:
   - **Preprocessing**: Generate `filtered_complaints.csv`:
     ```bash
     python src/preprocessing.py
     ```
   - **Embedding**: Create FAISS index and metadata:
     ```bash
     python src/embedding.py
     ```
   - **RAG Evaluation**: Generate evaluation table:
     ```bash
     python src/rag_pipeline.py
     ```
   - **Launch Interface**: Start the Gradio app:
     ```bash
     python src/app.py
     ```

6. **Compile the Report** (Optional):
   - Requires TeX Live with `texlive-full` and `texlive-fonts-extra`.
   - Compile `docs/report.tex` to generate `report.pdf`:
     ```bash
     latexmk -pdf docs/report.tex
     ```

## Task Descriptions
### Task 1: EDA and Preprocessing
- **Objective**: Analyze the CFPB dataset and preprocess it for embedding.
- **Implementation**:
  - Loaded `complaints.csv` using `pandas`.
  - Conducted EDA with `seaborn` and `matplotlib` to analyze product distribution and narrative lengths.
  - Filtered for five products and removed null narratives.
  - Cleaned narratives by lowercasing and removing special characters.
- **Deliverables**: `notebooks/eda_preprocessing.ipynb`, `src/preprocessing.py`, `data/filtered_complaints.csv`.
- **Challenges**: Empty output due to product name mismatches was resolved by adding debugging prints and normalizing product names.

### Task 2: Text Chunking and Embedding
- **Objective**: Chunk narratives, generate embeddings, and index them in a FAISS vector store.
- **Implementation**:
  - Used `langchain.text_splitter` with `chunk_size=200` and `chunk_overlap=20`.
  - Generated embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
  - Indexed embeddings using `faiss.IndexFlatL2`.
- **Deliverables**: `src/embedding.py`, `vector_store/complaint_index.faiss`, `vector_store/complaint_metadata.pkl`.
- **Challenges**: Empty chunks due to preprocessing issues were fixed with error handling and input validation.

### Task 3: RAG Pipeline and Evaluation
- **Objective**: Develop a RAG pipeline to retrieve relevant chunks and generate answers, then evaluate performance.
- **Implementation**:
  - Retrieved top-5 chunks using FAISS and `sentence-transformers`.
  - Generated answers with `gpt2` via `transformers.pipeline`.
  - Evaluated with five questions, producing a Markdown table.
- **Deliverables**: `src/rag_pipeline.py`, `docs/evaluation_table.md`.
- **Challenges**: Missing FAISS files, metadata mismatches, and `tabulate` dependency issues were resolved by re-running preprocessing, updating metadata keys, and installing `tabulate`.

### Task 4: Interactive Chat Interface
- **Objective**: Create a Gradio interface for non-technical users to query complaints.
- **Implementation**:
  - Built a `Blocks` interface with text input, submit/clear buttons, and output displays for answers and sources.
  - Integrated with `rag_pipeline.py` for query processing.
- **Deliverables**: `src/app.py`.
- **Challenges**: Ensured consistent file paths and error handling for seamless integration.

## Technical Report
A detailed report is available in `docs/report.pdf`, covering:
- Methodologies for each task (e.g., chunking strategy, LLM selection).
- Challenges and resolutions (e.g., empty dataset, LLM limitations).
- Evaluation results with sample answers and quality scores.
- Recommendations for future improvements (e.g., stronger LLM, streaming responses).

## Requirements
Key dependencies in `requirements.txt`:
- `pandas==2.2.2`
- `langchain==0.3.1`
- `sentence-transformers==3.1.1`
- `faiss-cpu==1.11.0`
- `transformers==4.53.1`
- `gradio==4.44.0`
- `tabulate==0.9.0`

## Notes
- Ensure `data/raw/complaints.csv` matches the expected format (e.g., correct product names).
- The `gpt2` model may produce generic answers; consider upgrading to a stronger LLM (e.g., via Hugging Face API) for production.
- Update `docs/evaluation_table.md` with actual answers and quality scores after running `rag_pipeline.py`.
- For LaTeX compilation, install TeX Live and required packages.

## Author
- Gedion Mekbeb Afework
- Date: July 2025
