---
marp: true
theme: default
paginate: true
style: |
  section { background: white; font-family: 'Inter', sans-serif; font-size: 28px; }
  h1 { color: #1e293b; border-bottom: 3px solid #f59e0b; font-size: 1.6em; margin-bottom: 0.5em; }
  h2 { color: #334155; font-size: 1.2em; margin: 0.5em 0; }
  code { background: #f8f9fa; font-size: 0.85em; font-family: 'Fira Code', monospace; border: 1px solid #e2e8f0; }
  pre { background: #f8f9fa; border-radius: 6px; padding: 1em; margin: 0.5em 0; }
  pre code { background: transparent; color: #1e293b; font-size: 0.7em; line-height: 1.5; }
  section { justify-content: flex-start; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# RAG & Vector Databases

**CS 203: Software Tools and Techniques for AI**
Prof. Nipun Batra, IIT Gandhinagar

---

# The LLM Knowledge Gap

**LLMs are frozen in time.**
- Trained on data up to a cutoff date (e.g., 2023).
- Don't know your private data (company emails, course syllabus).
- Can hallucinate facts.

**Solution: Retrieval Augmented Generation (RAG)**
- **Retrieve** relevant context from external source.
- **Augment** the prompt with this context.
- **Generate** answer using the augmented prompt.

---

# RAG Architecture

1.  **Ingestion**: Load documents → Split into chunks → Embed → Store in Vector DB.
2.  **Retrieval**: User Query → Embed Query → Find nearest neighbors in DB.
3.  **Generation**: Prompt = "Context: {retrieved_chunks} 
 Question: {query}" → LLM.

[Documents] --> (Embeddings) --> [Vector DB]
                                      ^
                                      |
[User Query] --> (Embeddings) --------+
                                      |
                                  [Top K Chunks]
                                      |
                                      v
[LLM] <----------------------- [Augmented Prompt]

---

# Embeddings: The Core Engine

**What is an embedding?**
- A vector (list of numbers) representing the *semantic meaning* of text.
- Similar texts have vectors close together in vector space.

**Models**:
- OpenAI `text-embedding-3-small` (1536 dim)
- Google `embedding-001`
- Open Source: `all-MiniLM-L6-v2` (Hugging Face)

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

emb1 = model.encode("The cat sits outside")
emb2 = model.encode("A man is playing guitar")
emb3 = model.encode("The feline rests outdoors")

# cosine_similarity(emb1, emb3) > cosine_similarity(emb1, emb2)
```

---

# Vector Databases

**Specialized databases for storing and searching high-dimensional vectors.**

**Why not standard SQL?**
- SQL is good for exact match (`WHERE id = 5`).
- Vector DB is good for approximate nearest neighbor (ANN) search.

**Popular Tools**:
- **ChromaDB**: Open-source, local/in-memory (Great for dev).
- **Pinecone**: Managed service (Scalable).
- **FAISS**: Facebook's library for dense retrieval (The engine behind many DBs).
- **Qdrant**: Rust-based, fast.
- **pgvector**: Postgres extension.

---

# ChromaDB Example

```python
import chromadb

# 1. Initialize Client
client = chromadb.Client()
collection = client.create_collection("course_docs")

# 2. Add Documents (Chroma handles embedding by default if not provided)
collection.add(
    documents=["CS203 covers AI tools.", "The exam is on Monday.", "Python is used."],
    metadatas=[{"source": "syllabus"}, {"source": "schedule"}, {"source": "intro"}],
    ids=["id1", "id2", "id3"]
)

# 3. Query
results = collection.query(
    query_texts=["When is the test?"],
    n_results=1
)

print(results['documents'])
# Output: [['The exam is on Monday.']]
```

---

# Building a RAG Pipeline: Step 1 (Ingestion)

**Chunking Matters**: LLMs have context limits, and we want precise retrieval.
- Split by character count?
- Split by paragraph?
- Recursive character text splitter (LangChain).

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "Long document..."
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(text)
# Now embed and store 'chunks'
```

---

# Building a RAG Pipeline: Step 2 (Retrieval)

```python
# User asks: "How do I install the tools?"
query_vector = embedding_model.encode("How do I install the tools?")

# Search Vector DB
results = collection.query(query_embeddings=[query_vector], n_results=3)
context_text = "\n".join(results['documents'][0])
```

---

# Building a RAG Pipeline: Step 3 (Generation)

```python
import google.generativeai as genai

prompt = f"""
You are a helpful teaching assistant. Answer the question based ONLY on the context below.

Context:
{context_text}

Question:
How do I install the tools?
"""

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content(prompt)
print(response.text)
```

---

# Orchestration Frameworks

Writing all this glue code is tedious. Frameworks help:

**LangChain**:
- Massive ecosystem.
- Chains, Agents, Integrations.
- Can be complex/verbose.

**LlamaIndex**:
- Specialized for data ingestion/retrieval.
- better for complex data structures (hierarchical indices).

**Haystack**:
- Pipeline-centric, robust.

---

# LangChain Example

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Setup
db = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings())
retriever = db.as_retriever()
llm = OpenAI()

# Chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever
)

# Run
print(qa.run("What is the grading policy?"))
```

---

# Advanced RAG Techniques

**1. Hybrid Search**:
- Combine Vector Search (semantic) + Keyword Search (BM25).
- Good for exact terms like product IDs or names.

**2. Re-ranking**:
- Retrieve 50 docs quickly (Vector DB).
- Re-rank top 50 using a slower, more accurate model (Cross-Encoder).
- Pass top 5 to LLM.

**3. Query Expansion**:
- LLM rewrites user query into multiple versions.
- Search all versions, deduplicate results.

**4. Metadata Filtering**:
- `WHERE year = 2024 AND embedding_similarity > 0.8`.

---

# Lab: Chat with Your PDF

**Objective**: Build a tool to upload a PDF and ask questions about it.

**Tools**:
- **pypdf**: Extract text.
- **RecursiveCharacterTextSplitter**: Chunking.
- **ChromaDB**: Vector Store.
- **Gemini/OpenAI API**: Embeddings & Generation.
- **Streamlit**: UI.

**Workflow**:
1.  User uploads `paper.pdf`.
2.  App extracts text -> chunks -> embeds -> stores in session ChromaDB.
3.  User types "What is the main contribution?".
4.  App retrieves chunks -> generates answer.

---

# Resources

- **Pinecone Learning Center**: pinecone.io/learn
- **LangChain Docs**: python.langchain.com
- **ChromaDB**: trychroma.com
- **DeepLearning.AI Short Courses**: "Building Systems with LLM API"

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Questions?

