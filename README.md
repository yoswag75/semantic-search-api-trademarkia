# Semantic Search + Cluster-Aware Semantic Cache API

A high‑performance semantic search and intelligent semantic caching
system built from first principles using FastAPI, FAISS, and
probabilistic clustering.

This system demonstrates how vector search infrastructure, latent
semantic clustering, and cache-aware routing algorithms can be combined
to build an efficient semantic retrieval layer without relying on
external caching systems such as Redis or Memcached.

------------------------------------------------------------------------

# System Architecture

Client Query │ ▼ FastAPI Router │ ▼ Query Embedding
(SentenceTransformers) │ ▼ Cluster Inference (PCA + GMM) │ ▼
Cluster‑Aware Semantic Cache │ ┌────┴───────────┐ │ │ Cache Hit Cache
Miss │ │ ▼ ▼ Return Result FAISS Vector Search │ ▼ Retrieve Document │ ▼
Cache Store

The architecture separates responsibilities across three layers:

Vector Engine -- Semantic representation and nearest neighbor retrieval\
Clustering Engine -- Latent topic discovery using probabilistic
modeling\
Semantic Cache -- Query-level caching optimized with cluster-aware
routing

------------------------------------------------------------------------

# Dataset

The system operates on the 20 Newsgroups dataset (\~20,000 documents).

To ensure clean semantic signals, preprocessing removed:

-   Email headers
-   Footers
-   Quoted reply chains

This prevents target leakage and isolates the raw semantic content of
each document.

------------------------------------------------------------------------

# Component 1: Vector Engine

Embedding model: all-MiniLM-L6-v2 from SentenceTransformers.

Properties:

Embedding dimension: 384\
Model size: \~80MB\
Inference: CPU optimized\
Latency: \~2--4ms per query

This model offers strong semantic quality with extremely fast CPU
inference.

## Vector Index

The FAISS index used is IndexFlatIP.

Cosine similarity can be rewritten as:

cos(A,B) = (A·B) / (\|\|A\|\| \|\|B\|\|)

By L2 normalizing vectors before indexing, cosine similarity becomes an
inner product search.

This enables exact cosine similarity retrieval using FAISS's optimized
inner‑product kernels.

Complexity:

Index Build: O(Nd)\
Query Search: O(Nd)\
Memory: O(Nd)

Where

N = number of documents (\~20k)\
d = embedding dimension (384)

At this scale, brute‑force search outperforms approximate nearest
neighbor methods due to SIMD vectorization.

------------------------------------------------------------------------

# Component 2: Fuzzy Semantic Clustering

Real‑world documents often belong to multiple topics.

Example:

"Gun control debate"

belongs simultaneously to:

-   Politics
-   Firearms
-   Law

Hard clustering like K‑Means cannot model this overlap effectively.

## Dimensionality Reduction

Original embeddings: 384D

To mitigate the curse of dimensionality:

PCA reduces embeddings to 50D.

Benefits:

-   Removes noise
-   Stabilizes covariance estimation
-   Improves clustering convergence

## Gaussian Mixture Model

Clustering uses a Gaussian Mixture Model (GMM).

Instead of hard assignments, GMM produces probability distributions:

Document → P(cluster1), P(cluster2), ...

Example:

Politics: 0.62\
Firearms: 0.28\
Law: 0.08

These soft assignments allow more accurate routing in the semantic
cache.

## Cluster Count

Although the dataset contains 20 labeled categories, the system uses:

K = 15 clusters

because several categories overlap semantically (e.g., Mac vs IBM
hardware).

Reducing cluster count produces cleaner latent structures.

------------------------------------------------------------------------

# Component 3: Cluster-Aware Semantic Cache

A custom semantic cache was implemented from scratch.

Each entry stores:

-   query
-   embedding
-   cluster distribution
-   result

## Cache Structure

Cache buckets are grouped by dominant cluster:

cache = { cluster_0: \[entries\], cluster_1: \[entries\], ... }

## Routing Algorithm

Instead of scanning the entire cache O(N):

1.  Compute query cluster probabilities
2.  Select clusters where probability \> 0.1
3.  Search only those cache buckets

This reduces expected lookup complexity to approximately:

O(N / K)

With K = 15 clusters, the search space is reduced roughly 15×.

## Similarity Threshold

Cosine similarity threshold:

0.88

Empirical observations:

> 0.95 → behaves like exact‑match cache\
> 0.88 → captures paraphrased intent\
> \<0.80 → starts matching unrelated topics

Thus 0.88 provides a safe semantic boundary.

------------------------------------------------------------------------

# API Endpoints

## POST /query

Request:

{ "query": "How do I install Linux on a new PC?" }

Response:

{ "query": "...", "cache_hit": true, "matched_query": "...",
"similarity_score": 0.91, "result": "...", "dominant_cluster": 4 }

## GET /cache/stats

Response:

{ "total_entries": 128, "hit_count": 87, "miss_count": 41, "hit_rate":
0.67 }

## DELETE /cache

Flushes the semantic cache.

------------------------------------------------------------------------

# Bootstrapping and Model Lifecycle

The application uses FastAPI's lifespan manager.

First startup performs:

1.  Dataset loading
2.  Embedding generation
3.  PCA training
4.  GMM clustering
5.  FAISS index creation

Artifacts are saved to:

.artifacts/

Stored files include:

embeddings.npy\
pca.pkl\
gmm.pkl\
faiss.index

Subsequent server restarts load these artifacts and start in \<1 second.

------------------------------------------------------------------------

# Running Locally

Setup:

python -m venv venv source venv/bin/activate pip install -r
requirements.txt

Run server:

uvicorn main:app --reload

API available at:

http://localhost:8000

Interactive docs:

http://localhost:8000/docs

------------------------------------------------------------------------

# Running with Docker

Start the system:

docker-compose up --build

The .artifacts directory is mounted as a volume to persist models across
container restarts.

------------------------------------------------------------------------

# Performance Characteristics

Corpus size: \~20,000 documents\
Embedding dimension: 384\
PCA dimension: 50\
Clusters: 15\
Cache lookup complexity: \~O(N/K)\
Query latency: \~10--30 ms

------------------------------------------------------------------------

# Design Philosophy

The system explores intent‑aware caching.

Traditional caches rely on exact keys, while semantic caching allows
queries like:

"How to install Linux?"\
"Steps to install Ubuntu?"

to reuse the same cached result.

This demonstrates how vector search systems can integrate with
intelligent caching strategies to build efficient semantic
infrastructure.

------------------------------------------------------------------------

# License

MIT
