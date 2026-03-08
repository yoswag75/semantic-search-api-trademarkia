import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger("SemanticSearch.Cache")

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.88, cluster_prob_threshold: float = 0.1):
        """
        The Semantic Cache Layer (Part 3).
        
        Tunable Decisions:
        1. similarity_threshold (0.88): 
        - This is the explicit value defining system behavior. 
        - At 0.95+, it's basically an exact-match cache (fails to catch paraphrases).
        - At <0.80, it starts exhibiting "semantic hallucination", grouping unrelated 
            queries just because they share a vague topic. 
        - 0.88 is chosen as the optimal boundary where distinct user intents are respected, 
            but minor rephrasings are caught.

        2. cluster_prob_threshold (0.1):
        - Determines which cache buckets to look in. By only scanning clusters where the 
            query has >10% probability, we skip evaluating against irrelevant cache entries,
            reducing lookup from O(N) to O(N/K) at scale.
        """
        self.similarity_threshold = similarity_threshold
        self.cluster_prob_threshold = cluster_prob_threshold
        
        # Cache structure: Dict mapping cluster_id to a list of cached entries.
        # This exploits the fuzzy clustering done in Part 2.
        self.store: Dict[int, List[dict]] = {}
        
        self.total_entries = 0
        self.hits = 0
        self.misses = 0

    def check(self, query_emb: np.ndarray, cluster_probs: np.ndarray) -> Tuple[bool, Optional[str], Optional[float], Optional[str]]:
        """
        Checks the cache using cluster-aware routing to avoid O(N) linear scans.
        Returns: (is_hit, matched_query_text, similarity_score, result_text)
        """
        # Identify viable clusters for this query based on the fuzzy distribution
        viable_clusters = np.where(cluster_probs > self.cluster_prob_threshold)[0]
        
        best_score = -1.0
        best_match = None

        for cluster_id in viable_clusters:
            if cluster_id not in self.store:
                continue
                
            for entry in self.store[cluster_id]:
                # Since vectors are normalized, dot product == cosine similarity
                score = np.dot(query_emb, entry["embedding"])
                
                if score > best_score:
                    best_score = float(score)
                    best_match = entry

        if best_match and best_score >= self.similarity_threshold:
            self.hits += 1
            logger.info(f"CACHE HIT! Score: {best_score:.3f}")
            return True, best_match["query"], best_score, best_match["result"]
            
        self.misses += 1
        return False, None, None, None

    def add(self, query_text: str, query_emb: np.ndarray, result_text: str, dominant_cluster: int):
        """
        Adds a missed query to the cache under its dominant cluster.
        """
        if dominant_cluster not in self.store:
            self.store[dominant_cluster] = []
            
        self.store[dominant_cluster].append({
            "query": query_text,
            "embedding": query_emb,
            "result": result_text
        })
        self.total_entries += 1

    def flush(self):
        """Resets the cache state completely."""
        self.store.clear()
        self.total_entries = 0
        self.hits = 0
        self.misses = 0