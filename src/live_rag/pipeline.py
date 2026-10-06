"""End-to-end RAG pipeline: fetch -> index -> retrieve -> augment -> generate."""

from dataclasses import dataclass

import pandas as pd
import pyterrier as pt

from .config import BASELINE_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT, Config
from .data import fetch_documents
from .generation import Generator
from .retrieval import BM25Retriever, build_index


@dataclass
class RAGResult:
    question: str
    baseline_answer: str
    rag_answer: str
    evidence: pd.DataFrame


class RAGPipeline:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.generator = Generator(self.config.model_id, self.config.max_new_tokens)
        self.retriever: BM25Retriever | None = None

    def refresh_index(self) -> int:
        """Pull the latest news and rebuild the BM25 index. Returns doc count."""
        docs = fetch_documents(self.config.rss_url)
        index_ref = build_index(docs, self.config.index_dir)
        self.retriever = BM25Retriever(index_ref, docs, self.config.top_k)
        return len(docs)

    @staticmethod
    def build_prompt(context: str, question: str) -> str:
        return f"Context:\n{context}\n\nQuestion:\n{question}"

    def ask(self, question: str) -> RAGResult:
        if self.retriever is None:
            self.refresh_index()

        evidence = self.retriever.search(question)
        context = "\n\n".join(evidence["text"])

        baseline = self.generator.generate(BASELINE_SYSTEM_PROMPT, question)
        grounded = self.generator.generate(
            RAG_SYSTEM_PROMPT, self.build_prompt(context, question)
        )
        return RAGResult(question, baseline, grounded, evidence)
