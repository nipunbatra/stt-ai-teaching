---
marp: true
theme: default
paginate: true
---

# Lab: RAG with Vector Databases

**Objective**: Build a "Chat with PDF" tool.

## Task 1: Data Ingestion
- Use `pypdf` to extract text from a PDF.
- Use `RecursiveCharacterTextSplitter` to chunk text.

## Task 2: Vector Store
- Initialize `ChromaDB` (persistent client).
- Embed chunks using `SentenceTransformer` or OpenAI/Gemini API.
- Add documents to collection.

## Task 3: Retrieval & Generation
- Query the database with a question.
- Retrieve top-k context.
- Pass context + question to LLM.
- Output answer.
