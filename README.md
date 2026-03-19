# Examples

Hands-on example projects for building AI-powered applications.

## Projects

### [RAG with MongoDB](./rag-mongodb/)

End-to-end workshop examples for building Retrieval-Augmented Generation (RAG)
applications using **MongoDB Atlas Vector Search** and **LangChain**.

| Example | Description |
|---|---|
| [01 — Basic RAG](./rag-mongodb/01-basic-rag/) | Ingest → embed → retrieve → generate pipeline |
| [02 — Chunking Strategies](./rag-mongodb/02-chunking-strategies/) | Fixed, recursive, sentence, and token-based chunking with comparison |
| [03 — Advanced RAG](./rag-mongodb/03-advanced-rag/) | Hybrid search, metadata filtering, multi-query, conversational RAG |
| [04 — Kubernetes](./rag-mongodb/04-kubernetes/) | Deploy MongoDB + RAG app as pods in Kubernetes |

**Stack:** Python · LangChain · MongoDB Atlas Vector Search · OpenAI · FastAPI · Kubernetes

**Quick start:**
```bash
cd rag-mongodb
pip install -r requirements.txt
cp .env.example .env      # fill in MONGODB_URI and OPENAI_API_KEY
cd 01-basic-rag && python main.py
```

**Kubernetes quick start:**
```bash
kubectl create namespace rag-workshop
# Edit rag-mongodb/04-kubernetes/secrets/rag-secrets.yaml with base64-encoded values
kubectl apply -k rag-mongodb/04-kubernetes/ -n rag-workshop
kubectl apply -f rag-mongodb/04-kubernetes/rag-app/job-ingest.yaml -n rag-workshop
kubectl port-forward svc/rag-app 8080:8080 -n rag-workshop
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Atlas Vector Search?"}'
```
