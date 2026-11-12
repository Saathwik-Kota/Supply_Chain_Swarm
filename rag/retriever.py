"""
Hybrid Dense + Sparse Vector Retrieval Engine (RAG) for NDRF & Logistics SOPs.

Combines TF-IDF lexical embeddings with Latent Semantic Analysis (TruncatedSVD)
dense projections and cosine similarity scoring to ground agent dispatch decisions
in verified NDMA disaster relief protocols and warehouse governance policies.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from config import RAG_TOP_K
from rag.knowledge_base import RELIEF_SOP_DOCUMENTS

logger = logging.getLogger(__name__)


class DisasterRAGRetriever:
    """In-memory hybrid sparse/dense vector store and semantic retriever."""

    def __init__(self, documents: Optional[List[Dict[str, str]]] = None):
        self.documents: List[Dict[str, str]] = list(documents or RELIEF_SOP_DOCUMENTS)
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._lsa_pipeline = None
        self._sparse_matrix = None
        self._dense_matrix = None
        self._build_index()

    def _doc_text(self, doc: Dict[str, str]) -> str:
        return (
            f"{doc.get('title', '')}. {doc.get('category', '')}. "
            f"{doc.get('disaster_types', '')}. {doc.get('content', '')}"
        )

    def _build_index(self) -> None:
        if not self.documents:
            return
        corpus = [self._doc_text(d) for d in self.documents]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self._vectorizer = vectorizer
        sparse_matrix = vectorizer.fit_transform(corpus)
        self._sparse_matrix = sparse_matrix

        n_docs, n_feats = sparse_matrix.shape
        n_components = max(2, min(10, n_docs - 1, n_feats - 1))
        if n_docs >= 3 and n_feats >= 3:
            lsa_pipeline = make_pipeline(
                TruncatedSVD(n_components=n_components, random_state=42),
                Normalizer(copy=False),
            )
            self._lsa_pipeline = lsa_pipeline
            self._dense_matrix = lsa_pipeline.fit_transform(sparse_matrix)
        else:
            self._lsa_pipeline = None
            self._dense_matrix = None

    def add_documents(self, new_docs: List[Dict[str, str]]) -> int:
        """Ingest additional SOP or historical report documents and rebuild index."""
        self.documents.extend(new_docs)
        self._build_index()
        return len(self.documents)

    def retrieve(
        self,
        query: str,
        disaster_type: Optional[str] = None,
        category_filter: Optional[str] = None,
        top_k: int = RAG_TOP_K,
    ) -> Dict[str, Any]:
        """
        Retrieve top-k relevant SOP chunks with hybrid semantic similarity scores.
        """
        start = time.time()
        if not self.documents or self._vectorizer is None:
            return {"query": query, "results": [], "latency_ms": 0.0}

        q_sparse = self._vectorizer.transform([query])
        sparse_sims = cosine_similarity(q_sparse, self._sparse_matrix).flatten()

        if self._lsa_pipeline is not None and self._dense_matrix is not None:
            q_dense = self._lsa_pipeline.transform(q_sparse)
            dense_sims = cosine_similarity(q_dense, self._dense_matrix).flatten()
            hybrid_scores = 0.65 * sparse_sims + 0.35 * np.maximum(0.0, dense_sims)
        else:
            hybrid_scores = sparse_sims

        dtype_norm = (disaster_type or "").lower().strip()
        scored_items = []
        for idx, doc in enumerate(self.documents):
            if category_filter and doc.get("category") != category_filter:
                continue

            score = float(hybrid_scores[idx])
            doc_dtypes = [d.strip().lower() for d in doc.get("disaster_types", "").split(",")]
            if dtype_norm and (dtype_norm in doc_dtypes or "all" in doc_dtypes):
                score += 0.08 if dtype_norm in doc_dtypes else 0.02

            scored_items.append((score, doc))

        scored_items.sort(key=lambda pair: pair[0], reverse=True)
        top_matches = [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "category": doc["category"],
                "source": doc["source"],
                "similarity_score": round(min(1.0, max(0.0, score)), 4),
                "content": doc["content"],
            }
            for score, doc in scored_items[:max(1, top_k)]
        ]

        latency_ms = round((time.time() - start) * 1000, 2)
        return {
            "query": query,
            "disaster_type_filter": disaster_type,
            "total_indexed_docs": len(self.documents),
            "results": top_matches,
            "latency_ms": latency_ms,
        }

    def get_index_stats(self) -> Dict[str, Any]:
        return {
            "document_count": len(self.documents),
            "vocab_size": len(self._vectorizer.vocabulary_) if self._vectorizer else 0,
            "embedding_dim": (
                int(self._dense_matrix.shape[1]) if self._dense_matrix is not None else 0
            ),
            "categories": sorted({d.get("category", "") for d in self.documents}),
        }


_GLOBAL_RAG_RETRIEVER: Optional[DisasterRAGRetriever] = None


def get_rag_retriever() -> DisasterRAGRetriever:
    global _GLOBAL_RAG_RETRIEVER
    if _GLOBAL_RAG_RETRIEVER is None:
        _GLOBAL_RAG_RETRIEVER = DisasterRAGRetriever()
    return _GLOBAL_RAG_RETRIEVER

