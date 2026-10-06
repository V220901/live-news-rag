"""Configuration for the live-news RAG pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    ticker: str = "TSLA"
    index_dir: str = "./tesla_news_index"
    top_k: int = 3
    model_id: str = "unsloth/Llama-3.2-3B-Instruct"
    max_new_tokens: int = 300

    @property
    def rss_url(self) -> str:
        return (
            "https://feeds.finance.yahoo.com/rss/2.0/headline"
            f"?s={self.ticker}&region=US&lang=en-US"
        )


BASELINE_SYSTEM_PROMPT = "You are a helpful financial assistant."

RAG_SYSTEM_PROMPT = (
    "You are a highly accurate financial analyst. "
    "You will be provided with live context from a news database. "
    "Answer the user's question ONLY using the provided context."
)
