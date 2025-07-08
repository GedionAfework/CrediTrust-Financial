# CrediTrust Financial Complaint Analysis

## Overview
This project develops an AI-powered complaint analysis system for CrediTrust Financial, a digital finance company operating in East African markets. The system transforms unstructured customer complaint data into actionable insights using a Retrieval-Augmented Generation (RAG) pipeline. This repository contains the implementation of **Task 1: Exploratory Data Analysis and Data Preprocessing** and **Task 2: Text Chunking, Embedding, and Vector Store Indexing**, leveraging the Consumer Financial Protection Bureau (CFPB) dataset.

The project aims to:
- Reduce the time for product managers to identify complaint trends from days to minutes.
- Enable non-technical teams (e.g., Support, Compliance) to query complaint data without data analysts.
- Shift CrediTrust from reactive to proactive issue resolution using real-time feedback.

## Project Structure
```
creditrust-complaint-analysis/
├── .venv/                           # Virtual environment
├── data/                            # Data files
│   ├── raw/                         # Raw CFPB dataset (complaints.csv)
│   └── filtered_complaints.csv      # Cleaned dataset from Task 1
├── notebooks/                       # Jupyter notebooks
│   └── eda_preprocessing.ipynb      # EDA and preprocessing for Task 1
├── src/                             # Source code
│   ├── preprocessing.py             # Data cleaning and filtering
│   ├── embedding.py                 # Text chunking, embedding, and indexing
├── vector_store/                    # FAISS vector store
│   ├── complaint_index.faiss        # FAISS index
│   └── complaint_metadata.pkl       # Metadata for embeddings
├── report.tex                       # LaTeX report source
├── metrics.json                     # Metrics for report
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Prerequisites
- Python 3.8+
- Git
- LaTeX distribution (e.g., TeX Live with `texlive-full`) for compiling the report
- CFPB dataset (`complaints.csv`) downloaded from [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/GedionAfework/CrediTrust-Financial.git
   cd creditrust-complaint-analysis
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/MacOS
   # Or: .venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download CFPB Dataset**:
   - Download `complaints.csv` from the CFPB website.
   - Place it in `data/raw/complaints.csv`.

## Usage
### Task 1: Exploratory Data Analysis and Preprocessing
- **Notebook**: Run `notebooks/eda_preprocessing.ipynb` to perform EDA and preprocess the dataset.
  ```bash
  jupyter notebook notebooks/eda_preprocessing.ipynb
  ```
- **Script**: Alternatively, run `src/preprocessing.py` to filter and clean the dataset.
  ```bash
  python src/preprocessing.py
  ```
- **Output**: Generates `data/filtered_complaints.csv` with complaints for Credit Card, Consumer Loan, Payday Loan, Checking or Savings Account, and Money Transfer, with cleaned narratives.

### Task 2: Text Chunking, Embedding, and Vector Store Indexing
- Run `src/embedding.py` to chunk narratives, generate embeddings, and create a FAISS vector store.
  ```bash
  python src/embedding.py
  ```
- **Output**: Generates `vector_store/complaint_index.faiss` and `vector_store/complaint_metadata.pkl`.

## Git Branches
- `main`: Initial setup and requirements.
- `task-1-eda-preprocessing`: Contains Task 1 deliverables (`eda_preprocessing.ipynb`, `preprocessing.py`, `filtered_complaints.csv`).
- `task-2-chunking-embedding`: Contains Task 2 deliverables (`embedding.py`, `vector_store/*`).

## Deliverables
### Task 1
- `notebooks/eda_preprocessing.ipynb`: EDA and preprocessing notebook.
- `src/preprocessing.py`: Modular preprocessing script.
- `data/filtered_complaints.csv`: Cleaned dataset.
- `report.tex`: Includes Task 1 summary (section updated with metrics).

### Task 2
- `src/embedding.py`: Script for chunking, embedding, and indexing.
- `vector_store/complaint_index.faiss`: FAISS vector store.
- `vector_store/complaint_metadata.pkl`: Metadata for embeddings.
- `report.tex`: Includes Task 2 summary (section updated with metrics).

## Metrics
Run `generate_metrics.py` to obtain detailed metrics, including:
- Task 1: Total complaints, narrative presence, narrative length statistics, product distribution, filtered dataset size.
- Task 2: Total chunks, chunks per complaint statistics, embedding dimension.

These metrics are included in `report.tex` for a comprehensive overview.

## Author
- **Name**: Gedion Mekbeb Afework
- **Date**: July 2025

## License
This project is for internal use at CrediTrust Financial and is not licensed for external distribution.
