# Dental Knowledge Assistant

A small Retrieval-Augmented Generation (RAG) application that answers dental hospital questions using information retrieved from a fictional hospital knowledge base.

This project was built as a hands-on AI engineering learning project to understand how an LLM can answer questions using supplied documents instead of relying only on its general knowledge.

> **Note:** Greenfield Dental Hospital and the documents in this repository are fictional and were created for learning purposes. This project is not intended to provide real medical advice.

## Live Demo

The application is deployed on Render:

https://dental-knowledge-assistant.onrender.com/

## What It Does

The assistant:

1. Loads hospital knowledge from text files.
2. Splits the documents into paragraph-level chunks.
3. Uses precomputed embeddings for the knowledge chunks.
4. Converts a user's question into an embedding through an embedding API.
5. Uses cosine similarity to compare the question embedding with the stored knowledge embeddings.
6. Retrieves the top matching chunks using Top-K retrieval.
7. Filters retrieved chunks using a similarity threshold.
8. Builds context from the sufficiently relevant chunks.
9. Supplies that context to an LLM.
10. Instructs the LLM to answer only from the supplied hospital context.
11. Displays the generated answer together with the source documents.

## RAG Pipeline

```text
Hospital Documents
        ↓
Document Loading
        ↓
Paragraph Chunking
        ↓
Precomputed Chunk Embeddings
        ↓
chunk_embeddings.json

User Question
        ↓
Question Embedding API
        ↓
Question Vector
        ↓
Cosine Similarity
        ↓
Top-K Retrieval
        ↓
Similarity Threshold
        ↓
Relevant Hospital Context
        ↓
Grounded LLM Generation
        ↓
Answer + Sources
```

## Knowledge Base

The fictional Greenfield Dental Hospital knowledge base currently contains:

- `appointment_policy.txt`
- `emergency_dental_care.txt`
- `post_extraction.txt`

These documents are intentionally small so the mechanics of RAG can be understood without introducing a vector database or RAG framework.

## How Retrieval Works

Each hospital document is divided into paragraph-level chunks.

The static knowledge chunks are converted into embeddings ahead of time and stored in:

```text
chunk_embeddings.json
```

When a user asks a question, only the question needs to be embedded.

The application then calculates cosine similarity between the question embedding and every stored knowledge embedding.

The results are ranked from most similar to least similar.

The system currently retrieves up to:

```text
Top-K = 5
```

A similarity threshold is then applied to the retrieved candidates:

```text
Similarity Threshold = 0.35
```

Only chunks that meet the threshold are supplied to the LLM.

If no retrieved chunks meet the threshold, the application reports that sufficiently relevant information could not be found in the hospital knowledge base.

## Example

Patient question:

```text
My tooth is bleeding, I had an extraction yesterday, what do I do?
```

The system:

```text
Question
   ↓
Embedding
   ↓
Similarity Comparison
   ↓
Relevant Hospital Chunks
   ↓
Grounded LLM Answer
```

Relevant information may be retrieved from the post-extraction and emergency dental care documents before the context is passed to the LLM.

For an unsupported question such as:

```text
How can I get my braces done?
```

the system can reject the question if the retrieved chunks do not meet the configured similarity threshold.

## Tech Stack

- Python
- Streamlit
- NumPy
- OpenAI-compatible Python SDK
- OpenRouter embedding API
- `liquid/lfm-2.5-embedding-350m:free`
- Groq API
- `openai/gpt-oss-20b`
- Render

The RAG pipeline is implemented directly without LangChain or a vector database so the underlying retrieval mechanics remain visible.

## Why Embeddings Are Precomputed

An earlier version of this project used a local SentenceTransformer model.

That version worked locally, but deploying it introduced much heavier dependencies, including PyTorch, and the embedding model failed to load reliably in the deployment environment.

The architecture was therefore changed.

Instead of loading a local embedding model whenever the application starts:

1. The fictional hospital knowledge chunks are embedded ahead of time.
2. Their embeddings are stored in `chunk_embeddings.json`.
3. The application loads those saved vectors when it starts.
4. Only new user questions are sent to the embedding API.

This keeps the deployed application lighter while preserving the core RAG retrieval process.

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

### 5. Configure Environment Variables

Create a `.env` file using `.env.example` as a guide:

```text
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Never commit your real `.env` file or API keys.

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application should open in your browser.

## Regenerating Knowledge Embeddings

The repository includes a script for generating the knowledge-base embeddings.

If the knowledge documents are changed, the stored embeddings should be regenerated so that they continue to represent the current document content.

Run:

```bash
python generate_embeddings.py
```

This generates the embeddings used by the retrieval system and stores them in:

```text
chunk_embeddings.json
```

Because embedding generation uses an external API, the required API key must be configured before running the script.

## Retrieval Evaluation

I manually tested the retrieval system using both supported and unsupported questions.

The evaluation included questions about:

- arriving early for an appointment
- arriving late for an appointment
- accident/emergency cases
- severe dental pain
- eating after extraction
- bleeding after extraction
- braces
- tooth whitening

During testing, I learned that retrieval quality cannot be judged only by whether the system returns a result.

The actual retrieved evidence also has to be inspected.

For example, the highest-scoring chunk is not always the chunk that most directly answers the question.

Increasing Top-K can allow a useful chunk to reach the context even when another semantically similar chunk ranks above it.

I also experimented with adding source metadata to the text used for generating embeddings.

This helped some queries, but it did not solve every retrieval problem.

## Similarity Threshold

The current similarity threshold is:

```text
0.35
```

This value was selected experimentally using a small manual evaluation set.

Supported questions generally produced noticeably stronger retrieval scores than deliberately unsupported questions in that test set.

However, `0.35` should not be treated as a universally correct threshold.

Changing the embedding model, knowledge base, chunking strategy, or question distribution could require recalibrating it.

## What I Learned

Building this project helped me understand:

- document loading
- paragraph-level chunking
- embeddings
- semantic similarity
- vector representations
- cosine similarity
- NumPy vector operations
- Top-K retrieval
- similarity thresholds
- retrieval filtering
- building context for an LLM
- grounding an LLM on retrieved documents
- source tracking
- separating retrieval from generation
- passing data between reusable Python functions
- API-based embeddings
- precomputing static embeddings
- Streamlit application development
- deployment debugging
- evaluating retrieval failures

One of the biggest lessons from this project was:

> **Successful retrieval does not automatically guarantee a grounded answer.**

A system can retrieve relevant information while the LLM still makes an inference that is stronger than what the source actually states.

This means RAG systems need evaluation at both levels:

```text
Did retrieval find the correct evidence?

AND

Did generation stay faithful to that evidence?
```

## Deployment Lessons

The first deployment attempt used a local SentenceTransformer embedding model.

The application worked locally, but the deployment environment struggled when loading the embedding model and its PyTorch dependencies.

To isolate the problem, I tested the application in smaller stages and confirmed that:

- Streamlit itself worked.
- The Render service worked.
- Importing the RAG application could work.
- Loading the local embedding model caused the deployment failure.

The final architecture replaced local embedding inference with API-based embeddings and precomputed vectors for the static knowledge base.

This reduced deployment requirements while keeping the retrieval mechanics implemented directly in Python.

## Known Limitations

This is a learning project, not a production clinical system.

Current limitations include:

- The knowledge base is small and fictional.
- Chunking is based on paragraphs rather than a production-grade chunking strategy.
- The `0.35` similarity threshold was selected from a small manual evaluation set and is not universally optimal.
- Top-K retrieval can still include irrelevant chunks.
- The highest similarity score does not always correspond to the most useful evidence.
- Even when correct evidence is retrieved, the LLM can occasionally make an inference stronger than what the source explicitly states.
- There is no vector database.
- There is no automated evaluation pipeline.
- There is no authentication or patient-data security layer.
- External API availability and rate limits can affect the application.
- The embedding API configuration used for this learning project should not be assumed suitable for real identifiable patient data.
- There is no production medical safety layer.

A real clinical system would require stronger validation, security, privacy controls, auditability, automated evaluation, human oversight, and clinical review.

## Safety

The assistant is instructed to:

- use only the retrieved hospital context
- avoid adding unsupported general knowledge
- avoid diagnosing patients
- state when the supplied context is insufficient
- avoid inferring policies that are not explicitly stated
- answer only the supported part when the context provides only a partial answer

The application is designed to demonstrate RAG mechanics and should not be used as a replacement for professional dental assessment or medical advice.

## Project Status

**v0.2 — Deployed RAG learning prototype**

Core RAG pipeline complete and deployed.

Built as part of my project-first AI engineering learning journey.