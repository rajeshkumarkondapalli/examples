# Example 4 — RAG Application on Kubernetes

Deploy the RAG application and MongoDB as pods in a Kubernetes cluster.
This example covers two deployment scenarios:

| Scenario | Description |
|---|---|
| **MongoDB as a Pod** | Single-node MongoDB pod (dev / testing) |
| **MongoDB via Atlas** | Production: use MongoDB Atlas, deploy only the RAG app pod |

## Directory Structure

```
04-kubernetes/
├── mongodb/              # MongoDB pod (dev/test only)
│   ├── pv.yaml           # PersistentVolume + PersistentVolumeClaim
│   ├── deployment.yaml   # MongoDB Deployment
│   └── service.yaml      # ClusterIP Service for MongoDB
├── rag-app/              # RAG application pod
│   ├── Dockerfile        # Container image for the RAG app
│   ├── deployment.yaml   # RAG app Deployment
│   ├── service.yaml      # Service (ClusterIP or LoadBalancer)
│   └── job-ingest.yaml   # One-shot Job to ingest documents
├── configmaps/
│   └── rag-config.yaml   # Non-sensitive configuration
├── secrets/
│   └── rag-secrets.yaml  # API keys and MongoDB URI (base64-encoded)
└── kustomization.yaml    # Kustomize overlay wiring everything together
```

## Prerequisites

- `kubectl` configured against a running cluster (local: minikube, kind, or k3s)
- Docker (to build the RAG app image)
- A namespace to deploy into (default: `rag-workshop`)

## Quick Start

### Option A — MongoDB as a Pod (dev/test)

```bash
# 1. Create namespace
kubectl create namespace rag-workshop

# 2. Create secrets (edit secrets/rag-secrets.yaml first)
kubectl apply -f secrets/rag-secrets.yaml -n rag-workshop

# 3. Deploy MongoDB
kubectl apply -f mongodb/ -n rag-workshop

# 4. Wait for MongoDB to be ready
kubectl rollout status deployment/mongodb -n rag-workshop

# 5. Build & push the RAG app image
docker build -t rag-app:latest -f rag-app/Dockerfile ../
# For minikube: eval $(minikube docker-env) before building

# 6. Apply ConfigMap
kubectl apply -f configmaps/rag-config.yaml -n rag-workshop

# 7. Deploy RAG app
kubectl apply -f rag-app/deployment.yaml -f rag-app/service.yaml -n rag-workshop

# 8. Run the ingest Job
kubectl apply -f rag-app/job-ingest.yaml -n rag-workshop
kubectl wait --for=condition=complete job/rag-ingest -n rag-workshop --timeout=300s

# 9. Check the service
kubectl port-forward svc/rag-app 8080:8080 -n rag-workshop
# Then visit http://localhost:8080
```

### Option B — MongoDB Atlas (production)

```bash
# Skip the mongodb/ manifests entirely.
# Set MONGODB_URI in secrets/rag-secrets.yaml to your Atlas connection string,
# then run steps 1, 2, 6–9 from Option A.
```

### Option C — Apply everything at once with Kustomize

```bash
# Edit secrets/rag-secrets.yaml with your values first
kubectl apply -k . -n rag-workshop
```

## Architecture

```
 ┌─────────────────────────────────────────────┐
 │              Kubernetes Cluster              │
 │                                             │
 │  ┌──────────────────┐  ┌─────────────────┐ │
 │  │   rag-app Pod    │  │  mongodb Pod    │ │
 │  │                  │  │                 │ │
 │  │  FastAPI / CLI   │──│  MongoDB 7.0    │ │
 │  │  LangChain       │  │  (dev/test)     │ │
 │  │  OpenAI Embed    │  │                 │ │
 │  └────────┬─────────┘  └─────────────────┘ │
 │           │ (production)                    │
 └───────────┼─────────────────────────────────┘
             │
             ▼ (Atlas connection string via Secret)
    ┌─────────────────────┐
    │  MongoDB Atlas       │
    │  (Vector Search)     │
    └─────────────────────┘
```

## Useful Commands

```bash
# View pod logs
kubectl logs -f deployment/rag-app -n rag-workshop

# Shell into the RAG app pod
kubectl exec -it deployment/rag-app -n rag-workshop -- /bin/bash

# Shell into the MongoDB pod
kubectl exec -it deployment/mongodb -n rag-workshop -- mongosh

# Scale the RAG app
kubectl scale deployment rag-app --replicas=3 -n rag-workshop

# Tear down everything
kubectl delete namespace rag-workshop
```
