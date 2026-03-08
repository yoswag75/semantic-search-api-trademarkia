import os
import json
import logging
import numpy as np
import faiss
import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("SemanticSearch.Engine")

ARTIFACTS_DIR = ".artifacts"
DOCS_PATH = os.path.join(ARTIFACTS_DIR, "docs.json")
EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
PCA_PATH = os.path.join(ARTIFACTS_DIR, "pca.joblib")
GMM_PATH = os.path.join(ARTIFACTS_DIR, "gmm.joblib")
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")

class SearchEngine:
    def __init__(self, n_clusters: int = 15):
        # We use a fast, highly capable embedding model.
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.n_clusters = n_clusters
        
        self.docs = []
        self.embeddings = None
        self.pca = None
        self.gmm = None
        self.index = None

    def _clean_data(self):
        """
        Fetches and cleans the dataset.
        Decision: We strip headers, footers, and quotes. 
        Why? Headers contain email addresses and explicit group names (target leakage).
        Quotes are repetitive reply-chains. Footers are uninformative signatures. 
        We want the raw semantic core of the user's message.
        """
        logger.info("Fetching 20 Newsgroups dataset...")
        raw_data = fetch_20newsgroups(
            subset='all', 
            remove=('headers', 'footers', 'quotes')
        ).data
        
        # Discard noise: Empty or very short documents lack semantic weight
        self.docs = [d for d in raw_data if len(d.strip()) > 50]
        logger.info(f"Retained {len(self.docs)} meaningful documents.")

    def _build_models(self):
        """Trains embeddings, PCA, GMM, and FAISS index."""
        logger.info("Computing embeddings (this may take a few minutes)...")
        # normalize_embeddings=True is required for Inner Product to act as Cosine Similarity
        self.embeddings = self.model.encode(self.docs, show_progress_bar=True, normalize_embeddings=True)

        logger.info("Fitting PCA (dimensionality reduction)...")
        # Why PCA? GMM struggles with high dimensional covariance matrices.
        self.pca = PCA(n_components=50, random_state=42)
        reduced_embs = self.pca.fit_transform(self.embeddings)

        logger.info(f"Fitting Gaussian Mixture Model (K={self.n_clusters})...")
        self.gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='full', random_state=42)
        self.gmm.fit(reduced_embs)

        logger.info("Building FAISS index for fast retrieval...")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        self._analyze_clusters(reduced_embs)

    def _analyze_clusters(self, reduced_embs):
        """
        Satisfies Part 2: Show boundary cases and cluster confidence.
        Demonstrates that soft-clustering is doing real work.
        """
        probs = self.gmm.predict_proba(reduced_embs)
        max_probs = np.max(probs, axis=1)
        
        # Find a highly confident document
        confident_idx = np.argmax(max_probs)
        # Find an uncertain/boundary document (highest probability is very low)
        uncertain_idx = np.argmin(max_probs)
        
        logger.info("--- Cluster Analysis ---")
        logger.info(f"Highly Confident Doc (Max Prob: {max_probs[confident_idx]:.4f})")
        logger.info(f"Snippet: {self.docs[confident_idx][:100]}...")
        
        logger.info(f"Boundary Doc (Max Prob: {max_probs[uncertain_idx]:.4f}) - Spans multiple clusters")
        logger.info(f"Snippet: {self.docs[uncertain_idx][:100]}...")
        logger.info(f"Distribution: {probs[uncertain_idx]}")
        logger.info("------------------------")

    def load_or_build(self):
        """Loads artifacts from disk if available, otherwise builds them."""
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        
        if all(os.path.exists(p) for p in [DOCS_PATH, EMBEDDINGS_PATH, PCA_PATH, GMM_PATH, INDEX_PATH]):
            logger.info("Loading pre-computed artifacts from disk...")
            with open(DOCS_PATH, "r") as f:
                self.docs = json.load(f)
            self.embeddings = np.load(EMBEDDINGS_PATH)
            self.pca = joblib.load(PCA_PATH)
            self.gmm = joblib.load(GMM_PATH)
            self.index = faiss.read_index(INDEX_PATH)
            logger.info("Artifacts loaded successfully.")
        else:
            logger.info("No artifacts found. Building from scratch...")
            self._clean_data()
            self._build_models()
            
            logger.info("Saving artifacts to disk...")
            with open(DOCS_PATH, "w") as f:
                json.dump(self.docs, f)
            np.save(EMBEDDINGS_PATH, self.embeddings)
            joblib.dump(self.pca, PCA_PATH)
            joblib.dump(self.gmm, GMM_PATH)
            faiss.write_index(self.index, INDEX_PATH)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)[0]

    def get_cluster_distribution(self, query_emb: np.ndarray) -> np.ndarray:
        reduced = self.pca.transform([query_emb])
        return self.gmm.predict_proba(reduced)[0]

    def search_corpus(self, query_emb: np.ndarray) -> str:
        """Finds the most semantically relevant document in the corpus."""
        # FAISS search expects 2D array
        query_emb_2d = query_emb.reshape(1, -1)
        distances, indices = self.index.search(query_emb_2d, 1)
        best_idx = indices[0][0]
        return self.docs[best_idx]