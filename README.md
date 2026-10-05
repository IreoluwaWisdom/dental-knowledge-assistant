# Dental Knowledge Assistant

A small Retrieval-Augmented Generation (RAG) project that answers dental hospital questions using information retrieved from a fictional hospital knowledge base.

This project was built as a hands-on AI engineering learning project to understand how an LLM can answer questions using supplied documents instead of relying only on its general knowledge.

> **Note:** Greenfield Dental Hospital and the documents in this repository are fictional and were created for learning purposes. This project is not intended to provide real medical advice.

## What It Does

The assistant:

1. Loads hospital knowledge from text files.
2. Splits the documents into smaller chunks.
3. Converts those chunks into embeddings.
4. Converts a patient's question into an embedding.
5. Uses cosine similarity to find relevant chunks.
6. Retrieves the top matching chunks.
7. Rejects questions when the best retrieval score is below a similarity threshold.
8. Supplies the retrieved context to an LLM.
9. Instructs the LLM to answer only from the supplied hospital context.

## RAG Pipeline

```text
Hospital Documents
        ↓
Document Loading
        ↓
Chunking
        ↓
Embeddings
        ↓
Patient Question
        ↓
Question Embedding
        ↓
Cosine Similarity
        ↓
Top-K Retrieval
        ↓
Similarity Threshold
        ↓
Retrieved Context
        ↓
LLM
        ↓
Grounded Answer
```

## Knowledge Base

The fictional Greenfield Dental Hospital knowledge base currently contains:

- `appointment_policy.txt`
- `emergency_dental_care.txt`
- `post_extraction.txt`

These documents are intentionally small so the mechanics of RAG can be understood without introducing a vector database or RAG framework.

## Example

Patient question:

```text
My tooth is bleeding, I had an extraction yesterday, what do I do?
```

The system retrieves relevant information from the post-extraction and emergency dental care documents before sending the retrieved context to the LLM.

For an unsupported question such as:

```text
How can I get my braces done?
```

the system can reject the question when the retrieval similarity is below the configured threshold.

## Tech Stack

- Python
- Sentence Transformers
- `all-MiniLM-L6-v2` embedding model
- Cosine similarity
- Groq through the OpenAI-compatible Python SDK
- `openai/gpt-oss-20b`

The RAG pipeline is implemented directly without LangChain or a vector database so the underlying mechanics remain visible.

## Running Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd dental-knowledge-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the API key

Create a `.env` file using `.env.example` as a guide:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Never commit your real `.env` file.

### 6. Run

```bash
python rag.py
```

## What I Learned

Building this project helped me understand:

- document loading and chunking
- embeddings and semantic similarity
- cosine similarity
- top-k retrieval
- similarity thresholds
- building context for an LLM
- grounding an LLM on retrieved documents
- handling questions that are not covered by the knowledge base
- separating retrieval from generation
- passing data between reusable Python functions
- debugging retrieval failures

One important lesson was that successful retrieval does not automatically guarantee a perfectly grounded answer.

## Known Limitations

This is a learning project, not a production clinical system.

Current limitations include:

- The knowledge base is small and fictional.
- Chunking is based on paragraphs rather than a production-grade chunking strategy.
- The similarity threshold (`0.45`) was selected experimentally from a small set of test questions and is not universally optimal.
- Top-k retrieval can sometimes include irrelevant chunks.
- Even when the correct evidence is retrieved, the LLM can occasionally make an inference stronger than what the source explicitly states.
- There is no vector database.
- There is no automated evaluation system.
- There is no production medical safety layer.

A real clinical system would require stronger validation, evaluation, security, auditability, human oversight, and clinical review.

## Safety

The assistant is instructed not to diagnose patients and to answer only from retrieved context.

It is designed to demonstrate RAG mechanics and should not be used as a replacement for professional dental assessment.

## Project Status

**v0.1 — RAG learning prototype**