# RAG with MongoDB — Workshop Examples

Hands-on examples for building, optimizing, and deploying Retrieval-Augmented Generation (RAG)
applications using **MongoDB Atlas Vector Search** and **LangChain**.

## Workshop Goals

- Identify the architecture of a RAG application
- Describe and implement chunking strategies for data preparation
- Build a full RAG app powered by LangChain + MongoDB
- Deploy the application and MongoDB in Kubernetes

## Project Structure

```
rag-mongodb/
├── 01-basic-rag/           # End-to-end RAG pipeline
├── 02-chunking-strategies/ # Data preparation & chunking approaches
├── 03-advanced-rag/        # Hybrid search, metadata filtering, re-ranking, chat
├── 04-kubernetes/          # Kubernetes manifests: MongoDB pod + RAG app pod
├── shared/                 # Shared utilities and sample data
├── app.py                  # FastAPI server (used by the Docker container)
├── requirements.txt        # Python dependencies
└── .env.example            # Environment variable template
```

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| MongoDB Atlas account | Free tier works for examples 1–3 |
| OpenAI API key | Required for embeddings and LLM |
| kubectl + cluster | Required for example 4 |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env: add MONGODB_URI and OPENAI_API_KEY

# 3. Run the basic RAG demo
cd 01-basic-rag && python main.py
```

## Environment Variables

| Variable | Description |
|---|---|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `OPENAI_API_KEY` | OpenAI API key for embeddings + LLM |
| `DB_NAME` | Database name (default: `rag_workshop`) |
| `COLLECTION_NAME` | Collection name (default: `documents`) |

## Examples at a Glance

### [01 — Basic RAG](./01-basic-rag/)
Complete ingest → retrieve → generate pipeline.

```python
# Ingest documents into MongoDB Atlas
python 01-basic-rag/ingest.py

# Ask a question
chain.invoke("What is MongoDB Atlas Vector Search?")
```

### [02 — Chunking Strategies](./02-chunking-strategies/)
Compare four chunking approaches and understand their trade-offs.

```python
# See stats for all strategies on a sample text
python 02-chunking-strategies/compare.py
```

### [03 — Advanced RAG](./03-advanced-rag/)
Production patterns: hybrid search, metadata filtering, multi-query, conversational chat.

```python
# Metadata-filtered retrieval
retriever = build_filtered_retriever(category="database", k=3)

# Multi-turn conversation
chain.invoke({"question": "What distance metrics does it support?"}, ...)
```

### [04 — Kubernetes](./04-kubernetes/)
Deploy MongoDB as a pod and the RAG app in a Kubernetes cluster.

```bash
kubectl create namespace rag-workshop
kubectl apply -k 04-kubernetes/ -n rag-workshop
kubectl port-forward svc/rag-app 8080:8080 -n rag-workshop
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Atlas Vector Search?"}'
```

## Recommended Learning Order

Work through the examples in sequence. Each one builds on the previous.

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1 — 01-basic-rag                               ~30 min        │
│                                                                     │
│  Goal: Get one complete RAG loop working end-to-end                 │
│  You will: set up Atlas, embed 6 docs, ask 4 questions              │
│  Stop here if: Atlas index errors, auth failures, missing .env      │
│  Do NOT continue until main.py produces answers                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ ✓ main.py works
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2 — 02-chunking-strategies                     ~20 min        │
│                                                                     │
│  Goal: Understand how document splitting affects retrieval          │
│  Start with: compare.py (offline, no Atlas/OpenAI needed)          │
│  Then: ingest_with_chunks.py to see differences in real retrieval   │
│  Key insight: chunk quality matters more than chunk count           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ ✓ you can explain why recursive > fixed
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3 — 03-advanced-rag                            ~45 min        │
│                                                                     │
│  Run scripts in this order:                                         │
│    1. filtered_retrieval.py  (cheapest, no extra setup)             │
│    2. multi_query.py         (see multi-LLM-call pattern)           │
│    3. hybrid_search.py       (needs fulltext_index built first)     │
│    4. conversational_rag.py  (most complex, needs history chain)    │
│  Read 03-advanced-rag/NOTES.md before each script                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ ✓ conversational chain answers follow-ups
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4 — 04-kubernetes                              ~60 min        │
│                                                                     │
│  Goal: Run the RAG app as pods in a cluster                         │
│  Start with: read NOTES.md top to bottom first                      │
│  Prerequisite: Minikube or equivalent cluster running               │
│  Run: deploy MongoDB pod → ingest Job → query via port-forward      │
└─────────────────────────────────────────────────────────────────────┘
```

**Each folder has a `NOTES.md` — read it before running any script in that folder.**

---

## Common Mistakes to Avoid

### Setup

| Mistake | What Happens | What to Do Instead |
|---|---|---|
| Running `main.py` before Atlas index is Active | `OperationFailure: Index not found` | Wait 60s, check Atlas UI: Search → vector_index → Status: Active |
| Querying with wrong collection name | Returns 0 results silently | `.env` `COLLECTION_NAME` must match what `ingest.py` created |
| Forgetting to whitelist your IP | `connection timed out` | Atlas → Network Access → Add IP Address |
| Sharing `.env` or committing it to git | API key leak | `.env` is already in `.gitignore`; never override this |
| Running `ingest.py` repeatedly without clearing | Duplicate documents pile up | `ingest.py` deletes existing docs first — this is intentional |

### Embeddings and Vectors

| Mistake | What Happens | What to Do Instead |
|---|---|---|
| Mixing embedding models between ingest and query | Garbage retrieval, wrong dimensions | Always use the same model in both `ingest.py` and retrieval |
| Changing `chunk_size` after ingest without re-ingesting | New chunks don't match existing index | Re-run `ingest.py` after changing any chunking parameter |
| Setting `k` larger than documents in DB | No error, but fewer results returned | Keep `k` ≤ number of documents in collection |
| Using dot_product distance without normalizing vectors | Wrong ranking | Use `cosine` unless you know your vectors are unit-normalized |

### Advanced RAG (Example 3)

| Mistake | What Happens | What to Do Instead |
|---|---|---|
| Querying before `fulltext_index` is Active | `hybrid_search` returns only vector results or errors | Wait for both indexes in Atlas UI before querying |
| Forgetting `pre_filter` field isn't in the index definition | Filter silently returns 0 results | Re-create the vector index with filter field paths included |
| Passing raw follow-up questions to the vector store | Pronoun "it" finds nothing relevant | The condensation step in `conversational_rag.py` fixes this — don't bypass it |
| Expecting in-memory chat history to survive restarts | History lost; follow-ups lose context | Replace `ChatMessageHistory` with a MongoDB-backed store for production |

### Kubernetes (Example 4)

| Mistake | What Happens | What to Do Instead |
|---|---|---|
| Building Docker image outside Minikube's Docker context | `ImagePullBackOff` | Run `eval $(minikube docker-env)` before `docker build` |
| Applying Deployment before Secret | Pod fails with missing env var | Apply secrets first, then deployments |
| Using `hostPath` PV on multi-node clusters | Pod can't find the volume on another node | Use a cloud StorageClass (EBS, GCE PD) for multi-node |
| Committing `secrets/rag-secrets.yaml` with real values | Credential exposure | Add to `.gitignore`; use external secrets manager in production |
| Skipping the ingest Job | App returns no results | Run `job-ingest.yaml` once after deployment, verify with `kubectl logs` |

---

## Quick Reference Card

### Commands You Will Use Most

```bash
# --- Setup ---
pip install -r requirements.txt
cp .env.example .env

# --- Example 1 ---
cd 01-basic-rag
python main.py                          # ingest + wait + query (all-in-one)
python ingest.py                        # ingest only
python -c "from rag_chain import build_rag_chain; ..."  # query only

# --- Example 2 ---
cd 02-chunking-strategies
python compare.py                       # offline, no Atlas/OpenAI needed
python ingest_with_chunks.py            # ingest with chosen strategy
python main.py                          # compare strategies + query

# --- Example 3 ---
cd 03-advanced-rag
python filtered_retrieval.py            # filter by category/source
python multi_query.py                   # auto-expand query variants
python hybrid_search.py                 # vector + full-text fusion
python conversational_rag.py            # multi-turn conversation
python main.py                          # run all four patterns

# --- Example 4 ---
eval $(minikube docker-env)             # point Docker at Minikube
docker build -t rag-app:latest -f 04-kubernetes/rag-app/Dockerfile .
kubectl create namespace rag-workshop
kubectl apply -k 04-kubernetes/ -n rag-workshop
kubectl rollout status deployment/rag-app -n rag-workshop
kubectl apply -f 04-kubernetes/rag-app/job-ingest.yaml -n rag-workshop
kubectl port-forward svc/rag-app 8080:8080 -n rag-workshop
curl -X POST http://localhost:8080/ask -H "Content-Type: application/json" \
  -d '{"question": "What is Atlas Vector Search?"}'
```

### Atlas Index Status Check

```bash
# Via Atlas UI:
#   Your Cluster → Search → vector_index → Status: Active
#
# Via Python:
python -c "
from pymongo import MongoClient
import os; from dotenv import load_dotenv; load_dotenv()
c = MongoClient(os.getenv('MONGODB_URI'))
db = c[os.getenv('DB_NAME','rag_workshop')]
indexes = list(db[os.getenv('COLLECTION_NAME','documents')].list_search_indexes())
for i in indexes: print(i['name'], '->', i['status'])
"
```

### Environment Variables at a Glance

```bash
MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?retryWrites=true&w=majority
OPENAI_API_KEY=sk-proj-...
DB_NAME=rag_workshop          # default — change if you want a different DB
COLLECTION_NAME=documents     # default — change if you want a different collection
```

### Key Numbers to Remember

| Parameter | Default | When to Change |
|---|---|---|
| Atlas index build wait | 60 s | Increase to 90 s on slow clusters |
| chunk_size | 500 chars | Decrease for precise answers; increase for more context |
| chunk_overlap | 100 chars | Keep at ~20% of chunk_size |
| retrieval k | 3 docs | Increase if answers feel incomplete; costs more tokens |
| multi-query variants | 3 | Increase for broader recall; each adds one LLM call |
| Kubernetes replicas | 2 | Scale up for load; scale down to 1 for dev |

### Troubleshooting One-Liners

```bash
# Test Atlas connectivity
python -c "from pymongo import MongoClient; import os; from dotenv import load_dotenv; load_dotenv(); MongoClient(os.getenv('MONGODB_URI')).admin.command('ping'); print('Atlas OK')"

# Count documents in collection
python -c "from pymongo import MongoClient; import os; from dotenv import load_dotenv; load_dotenv(); c=MongoClient(os.getenv('MONGODB_URI')); print(c[os.getenv('DB_NAME','rag_workshop')][os.getenv('COLLECTION_NAME','documents')].count_documents({}),'documents')"

# Test OpenAI key
python -c "from openai import OpenAI; import os; from dotenv import load_dotenv; load_dotenv(); r=OpenAI().embeddings.create(input='test',model='text-embedding-3-small'); print('OpenAI OK, dims:', len(r.data[0].embedding))"

# Kubernetes: see all resources
kubectl get all -n rag-workshop

# Kubernetes: tail app logs live
kubectl logs -f deployment/rag-app -n rag-workshop
```

---

## RAG Architecture Overview

```
  User Query
      │
      ▼
┌─────────────┐     ┌──────────────────────┐
│  Embedding  │────▶│  MongoDB Atlas       │
│  Model      │     │  Vector Search       │
└─────────────┘     └──────────┬───────────┘
                               │ Top-K docs
                               ▼
                    ┌──────────────────────┐
                    │  LLM (GPT-4o-mini)   │
                    │  + Retrieved Context │
                    └──────────┬───────────┘
                               │
                               ▼
                          Final Answer
```
