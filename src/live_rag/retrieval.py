"""PyTerrier indexing and BM25 retrieval."""

import re

import pandas as pd
import pyterrier as pt


def build_index(docs: pd.DataFrame, index_dir: str):
    """Index documents with PyTerrier and return the index reference."""
    indexer = pt.terrier.IterDictIndexer(index_dir, overwrite=True)
    return indexer.index(docs.to_dict("records"))


def _clean_query(query: str) -> str:
    # Terrier's query parser chokes on punctuation such as ( ) ' ?
    return re.sub(r"[^\w\s]", " ", query).strip()


class BM25Retriever:
    def __init__(self, index_ref, docs: pd.DataFrame, top_k: int = 3):
        self.docs = docs
        self.top_k = top_k
        self.bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25", num_results=top_k)

    def search(self, query: str) -> pd.DataFrame:
        """Return the top-k documents (with scores) for a query."""
        results = self.bm25.search(_clean_query(query))
        return results.merge(self.docs, on="docno").head(self.top_k)
