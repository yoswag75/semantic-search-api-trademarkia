import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
import numpy as np

from schemas import QueryRequest, QueryResponse, CacheStatsResponse
from engine import SearchEngine
from semantic_cache import SemanticCache

# Configure logging for professional output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SemanticSearch.API")

# Global instances
engine = SearchEngine(n_clusters=15)
cache = SemanticCache(similarity_threshold=0.88, cluster_prob_threshold=0.1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager. Ensures the ML engine, vector DB, and models
    are loaded before the API starts accepting traffic.
    """
    logger.info("Initializing Service...")
    engine.load_or_build()
    logger.info("Service Initialization Complete. Ready for requests.")
    yield
    # Teardown logic if needed goes here

app = FastAPI(
    title="Semantic Cache & Search API",
    description="A lightweight, Redis-free semantic cache using fuzzy clustering on the 20 Newsgroups dataset.",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/query", response_model=QueryResponse)
async def process_query(req: QueryRequest):
    # 1. Embed the query
    query_emb = engine.embed_query(req.query)
    
    # 2. Get the fuzzy cluster distribution (Part 2)
    cluster_probs = engine.get_cluster_distribution(query_emb)
    dominant_cluster = int(np.argmax(cluster_probs))
    
    # 3. Check the Semantic Cache (Part 3)
    is_hit, matched_q, score, cached_result = cache.check(query_emb, cluster_probs)
    
    if is_hit:
        return QueryResponse(
            query=req.query,
            cache_hit=True,
            matched_query=matched_q,
            similarity_score=score,
            result=cached_result,
            dominant_cluster=dominant_cluster
        )
        
    # 4. On Cache Miss: Compute result from Vector DB
    corpus_result = engine.search_corpus(query_emb)
    
    # 5. Store in Cache for future
    cache.add(
        query_text=req.query,
        query_emb=query_emb,
        result_text=corpus_result,
        dominant_cluster=dominant_cluster
    )
    
    return QueryResponse(
        query=req.query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=corpus_result,
        dominant_cluster=dominant_cluster
    )

@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    total_requests = cache.hits + cache.misses
    hit_rate = (cache.hits / total_requests) if total_requests > 0 else 0.0
    
    return CacheStatsResponse(
        total_entries=cache.total_entries,
        hit_count=cache.hits,
        miss_count=cache.misses,
        hit_rate=round(hit_rate, 3)
    )

@app.delete("/cache")
async def flush_cache():
    cache.flush()
    return {"message": "Cache flushed successfully"}