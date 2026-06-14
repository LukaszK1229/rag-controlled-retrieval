# Controlled Retrieval RAG API

## Overview

Controlled Retrieval RAG API is a Retrieval-Augmented Generation (RAG) system designed to answer questions using an internal document knowledge base while enforcing retrieval quality controls.

The project combines:

* FastAPI
* FAISS
* SQLite
* OpenRouter
* Docker
* Google Cloud Run

Unlike basic RAG implementations, this system includes metadata filtering, retrieval confidence checks, structured logging, and observability features designed to improve reliability and reduce hallucinations.

---

## Problem Statement

Traditional RAG systems often generate answers even when retrieval quality is poor.

This project introduces a control layer between retrieval and generation.

Before an answer is generated, the system evaluates retrieval confidence using configurable thresholds. If confidence is too low, the response is blocked instead of generating a potentially incorrect answer.

---

## Architecture

User Question

↓

Embedding Generation

(OpenRouter Embeddings API)

↓

FAISS Retrieval

(IndexFlatL2)

↓

Metadata Filtering

(SQLite)

↓

Confidence Gating

(Score Threshold + Gap Threshold)

↓

LLM Generation

(OpenRouter GPT-4.1 Mini)

↓

Response

---

## Key Features

### Semantic Search

Document chunks are embedded using:

* text-embedding-3-small

and stored inside a FAISS vector index.

### Metadata Filtering

Retrieved chunks must satisfy:

* Active document
* Active chunk
* Valid date range
* Allowed role scope

Only eligible chunks are passed further into the pipeline.

### Retrieval Confidence Controls

The system evaluates retrieval quality before calling the LLM.

Controls include:

* Score Threshold
* Score Gap Threshold

If retrieval confidence is insufficient:

```json
{
  "status": "stopped",
  "reason": "low_confidence"
}
```

is returned.

### Structured Logging

Every query is logged into SQLite.

Examples:

* query_received
* retrieval_filtered
* retrieval_results
* query_stopped
* answer_generated

This provides basic observability and retrieval debugging.

---

## Database Schema

### documents

Stores document-level metadata:

* doc_id
* source
* version
* doc_type
* role_scope
* valid_from
* valid_to
* is_active

### chunks

Stores chunked document content:

* chunk_id
* doc_id
* text
* version
* has_embedding
* is_active

### rag_logs

Stores runtime events:

* timestamp
* query_id
* event
* payload

---

## API

### Health Check

GET

```http
/health
```

Response:

```json
{
  "status": "ok"
}
```

---

### Query Endpoint

POST

```http
/query
```

Request:

```json
{
  "question": "Who approves emergency changes?"
}
```

Success Response:

```json
{
  "query_id": "12f362dd-10e9-436b-ba55-1318d9270b8e",
  "status": "ok",
  "answer": "Emergency changes require Ops Lead approval."
}

```
Example Request:

```json
{
  "question": "Where can I order the best pizza in New York?"
}
```

Stopped Response:

```json
{
  "query_id": "e7e66734-a563-4fba-9f27-fbaa480f7ad8",
  "status": "stopped",
  "reason": "low_confidence"
}
```

Example Response:

```json
{
  "query_id": "e7e66734-a563-4fba-9f27-fbaa480f7ad8",
  "status": "stopped",
  "reason": "low_confidence"
}
```

---

## Example Retrieval Control

### In-Scope Query

Question:

```text
Who approves emergency changes?
```

Top Retrieval Score:

```text
0.806
```

Result:

```text
Emergency changes require Ops Lead approval.
```

---

### Out-of-Scope Query

Question:

```text
Where can I order the best pizza in New York?
```

Top Retrieval Score:

```text
1.782
```

Result:

```text
STOP: low_confidence
```

The system refuses to answer because retrieval quality is insufficient.

---

## Deployment

The application is containerized using Docker and deployed to Google Cloud Run.

Deployment stack:

* Docker
* Artifact Registry
* Google Cloud Run

The API is publicly accessible through a Cloud Run endpoint.

---

## Tech Stack

Backend:

* FastAPI
* Python

Retrieval:

* FAISS
* SQLite

AI:

* OpenRouter
* GPT-4.1 Mini
* text-embedding-3-small

Infrastructure:

* Docker
* Google Cloud Run

Observability:

* SQLite Logs
* Structured Event Logging

---

## Future Improvements

Potential next steps:

* Source citations in API responses
* Authentication and rate limiting
* Cloud Logging integration
* Secret Manager integration
* CI/CD pipeline
* Hybrid retrieval (Vector + BM25)
* Re-ranking layer

---

## Purpose

This project was built as a practical exploration of Retrieval-Augmented Generation systems, retrieval quality controls, observability, and cloud deployment workflows.
