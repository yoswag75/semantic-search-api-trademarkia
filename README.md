```markdown
# Semantic Search + Cluster-Aware Semantic Cache API

A **high-performance semantic search and intelligent semantic caching system** built from first principles using **FastAPI**, **FAISS**, and **probabilistic clustering**.  

This system demonstrates how **vector search infrastructure**, **latent semantic clustering**, and **cache-aware routing algorithms** can be combined to build an efficient **semantic retrieval layer** without relying on external caching systems such as Redis or Memcached.

The repository is designed as a **systems-level demonstration of scalable semantic infrastructure**, focusing on:

- Algorithmic efficiency
- Architecture design
- Statistical modeling of text corpora
- Intelligent caching strategies

---

# System Architecture

The system is composed of **three tightly integrated subsystems**:

```

```
            ┌─────────────────────────────┐
            │        Client Query         │
            └──────────────┬──────────────┘
                           │
                           ▼
                ┌────────────────────┐
                │   FastAPI Router   │
                └─────────┬──────────┘
                          │
                          ▼
           ┌───────────────────────────┐
           │ Query Embedding Generator │
           │ SentenceTransformer       │
           └─────────┬─────────────────┘
                     │
                     ▼
      ┌──────────────────────────────────────┐
      │ Cluster Inference (PCA + GMM Model)  │
      │ Returns probability distribution     │
      └─────────────┬────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────┐
  │        Cluster-Aware Semantic Cache         │
  │                                             │
  │ Buckets grouped by dominant cluster IDs     │
  │ Only scans buckets where P(cluster) > 0.1   │
  └─────────────┬───────────────────────────────┘
                │
     Cache Hit  │  Cache Miss
                ▼
        ┌───────────────┐
        │ Return Result │
        └───────▲───────┘
                │
                ▼
      ┌─────────────────────┐
      │ FAISS Vector Search │
      │ Cosine Similarity   │
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │ Retrieve Document   │
      │ Store in Cache     │
      └─────────────────────┘
```

```

The architecture separates concerns across three layers:

| Layer | Responsibility |
|-----|-----|
| **Vector Engine** | Semantic representation + nearest neighbor retrieval |
| **Clustering Engine** | Latent topic discovery via probabilistic modeling |
| **Semantic Cache** | Query-level caching optimized using cluster routing |

---

# Dataset

The system operates on the **20 Newsgroups dataset (~20,000 documents)**.

To ensure **true semantic modeling**, the dataset was preprocessed by removing:

- Email headers
- Footers
- Quoted reply chains

This prevents **target leakage** and isolates the **core semantic content** of each document.

---

# Component 1: Vector Engine

## Embedding Model

The system uses:

```

all-MiniLM-L6-v2

```

from the `SentenceTransformers` library.

Reasons for this choice:

| Property | Value |
|------|------|
| Embedding dimension | 384 |
| Model size | ~80MB |
| Inference | CPU optimized |
| Latency | ~2–4 ms per query |

This model provides **strong semantic quality while maintaining extremely fast CPU inference**, making it suitable for **low-latency API systems**.

---

## Vector Index

Document embeddings are indexed using **FAISS (`faiss-cpu`)**.

```

IndexFlatIP

```

### Why Inner Product?

Cosine similarity can be rewritten as:

```

cos(A, B) = (A · B) / (||A|| ||B||)

```

By **L2-normalizing vectors before ingestion**, cosine similarity reduces to **pure inner product**.

This allows:

- **Exact cosine similarity search**
- Using FAISS's **highly optimized inner product kernels**

---

### Index Complexity

| Operation | Complexity |
|------|------|
| Index build | O(Nd) |
| Query search | O(Nd) |
| Memory | O(Nd) |

Where:

```

N = number of documents (~20k)
d = embedding dimension (384)

```

At this scale, **brute-force exact search is faster than approximate indexes** due to FAISS vectorization.

---

# Component 2: Fuzzy Semantic Clustering

A central design challenge is that **semantic topics overlap**.

Example:

```

"Gun control policy debate"

```

belongs simultaneously to:

- Politics
- Firearms
- Law

Traditional clustering (K-Means) produces **hard assignments**, which fails to capture this overlap.

---

## Dimensionality Reduction

The original embeddings are:

```

384 dimensions

```

High-dimensional clustering suffers from the **curse of dimensionality**, which destabilizes covariance estimation.

To mitigate this:

```

PCA → 384D → 50D

```

Benefits:

- Removes noise
- Stabilizes covariance matrices
- Improves clustering convergence

---

## Gaussian Mixture Model

Clustering is performed using a **Gaussian Mixture Model (GMM)**.

Instead of hard assignments:

```

Document → single cluster

```

GMM produces **probabilistic assignments**:

```

Document → P(cluster_1), P(cluster_2), ... P(cluster_k)

```

Example:

```

doc_134:
Politics: 0.62
Firearms: 0.28
Law: 0.08
Hardware: 0.02

```

This probabilistic representation enables **cluster-aware routing in the semantic cache**.

---

## Choice of Cluster Count

Although the dataset contains **20 labeled categories**, the model uses:

```

K = 15 clusters

```

Reason:

Empirical inspection revealed that several categories have **semantic overlap**, including:

- Mac vs IBM hardware
- Politics vs Firearms
- Religion vs Philosophy

Reducing to **15 clusters creates a cleaner latent representation**.

---

# Component 3: Cluster-Aware Semantic Cache

The system implements a **custom semantic cache from scratch**, without Redis or Memcached.

Each cache entry stores:

```

{
query,
embedding,
cluster_distribution,
result
}

```

---

## Cache Structure

The cache is organized into **cluster buckets**:

```

cache = {
cluster_0: [entry, entry],
cluster_1: [entry],
cluster_2: [],
...
}

```

Each query is assigned a **dominant cluster**.

---

## Routing Algorithm

Instead of scanning the entire cache:

```

O(N)

```

the system performs **cluster-aware routing**.

### Step 1 — Infer cluster probabilities

```

P(cluster_i | query)

```

### Step 2 — Select relevant clusters

Only clusters where:

```

P(cluster) > 0.1

```

are scanned.

### Step 3 — Perform cosine similarity search

Only within those buckets.

---

## Complexity Reduction

If:

```

N = total cache entries
K = clusters

```

Expected lookup complexity becomes approximately:

```

O(N / K)

```

For K=15, this results in **~15x reduction in scan space**.

---

## Similarity Threshold

The cache uses a **cosine similarity boundary**:

```

threshold = 0.88

```

This value was chosen empirically.

| Threshold | Behavior |
|------|------|
| >0.95 | Degenerates to exact-match cache |
| 0.88 | Captures paraphrased intent |
| <0.80 | Begins matching unrelated topics |

Examples of incorrect matches at low thresholds:

```

"installing Windows"
≈
"installing Linux"

````

Thus **0.88 acts as a safe semantic boundary**.

---

# API Endpoints

## POST `/query`

Execute a semantic search query with cache lookup.

### Request

```json
{
  "query": "How do I install Linux on a new PC?"
}
````

---

### Response

```json
{
  "query": "How do I install Linux on a new PC?",
  "cache_hit": true,
  "matched_query": "Steps to install Linux on a computer",
  "similarity_score": 0.913,
  "result": "Install Linux by creating a bootable USB...",
  "dominant_cluster": 4
}
```

---

## GET `/cache/stats`

Returns cache statistics.

### Response

```json
{
  "total_entries": 128,
  "hit_count": 87,
  "miss_count": 41,
  "hit_rate": 0.679
}
```

---

## DELETE `/cache`

Flushes all cached entries.

---

# Bootstrapping and Model Lifecycle

The system uses **FastAPI's lifespan manager** to handle model initialization.

On the **first server boot**, the system performs:

1. Dataset loading
2. Embedding generation for all documents
3. PCA model training
4. GMM clustering
5. FAISS index creation

Artifacts are saved to:

```
.artifacts/
```

Stored objects include:

```
embeddings.npy
pca.pkl
gmm.pkl
faiss.index
```

Subsequent server restarts **reuse the artifacts**, reducing startup time to:

```
< 1 second
```

---

# Running Locally

## Setup

```bash
git clone <repo>
cd semantic-cache-api

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

## Start the API

```bash
uvicorn main:app --reload
```

Server runs at:

```
http://localhost:8000
```

Interactive API documentation:

```
http://localhost:8000/docs
```

---

# Running with Docker

The repository includes a **Docker + docker-compose setup**.

Artifacts are persisted via a **mounted volume** so models are only built once.

---

## Start the system

```bash
docker-compose up --build
```

Volume mapping:

```
.artifacts/ → container:/app/.artifacts
```

This ensures:

* embeddings
* FAISS index
* PCA
* GMM models

persist across container restarts.

---

# Performance Characteristics

| Metric              | Value             |
| ------------------- | ----------------- |
| Corpus size         | ~20,000 documents |
| Embedding dimension | 384               |
| PCA dimension       | 50                |
| Clusters            | 15                |
| Cache lookup        | ~O(N/K)           |
| Query latency       | ~10–30 ms         |

---

# Key Design Decisions

### Why FAISS Flat Index?

At **20k vectors**, approximate indexes add unnecessary overhead.

Flat indexes leverage:

* SIMD vectorization
* optimized BLAS kernels

which outperform ANN structures at this scale.

---

### Why Probabilistic Clustering?

Soft clustering allows:

```
multi-topic documents
```

which is common in natural language corpora.

This improves **cache routing accuracy**.

---

### Why Build Cache from Scratch?

External caches:

* Redis
* Memcached

operate on **exact keys**, not **semantic intent**.

This project explores **intent-aware caching**, where queries like:

```
"How to install Linux?"
"Steps to install Ubuntu?"
```

can share the same cached response.

---

# Future Improvements

Potential extensions include:

* Hierarchical clustering
* Approximate FAISS indexes (HNSW, IVF)
* TTL-based cache eviction
* LRU cluster-aware eviction
* GPU embedding inference
* Distributed cache nodes


```
```
