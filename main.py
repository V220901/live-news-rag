"""Command-line entry point.

Usage:
    python main.py "What is driving Tesla's stock price this week?"
    python main.py --ticker NVDA --top-k 5 "Why is Nvidia moving today?"
"""

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from live_rag.config import Config  # noqa: E402
from live_rag.pipeline import RAGPipeline

DEFAULT_QUESTION = (
    "What specific news events or reports are driving Tesla's (TSLA) "
    "stock price this week?"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-news RAG demo (BM25 + Llama 3.2)")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--ticker", default=Config.ticker)
    parser.add_argument("--top-k", type=int, default=Config.top_k)
    parser.add_argument("--model", default=Config.model_id)
    args = parser.parse_args()

    config = dataclasses.replace(
        Config(), ticker=args.ticker, top_k=args.top_k, model_id=args.model
    )
    pipeline = RAGPipeline(config)
    n_docs = pipeline.refresh_index()
    print(f"Indexed {n_docs} live articles for {config.ticker}.\n")

    result = pipeline.ask(args.question)

    line = "=" * 80
    print(f"{line}\nQUESTION: {result.question}\n{line}")
    print(f"\nWITHOUT RAG:\n{result.baseline_answer}")
    print(f"\nWITH RAG:\n{result.rag_answer}")
    print(f"\n{line}\nEVIDENCE (BM25 top-{config.top_k})\n{line}")
    for rank, row in enumerate(result.evidence.itertuples(), start=1):
        print(f"[{rank}] score={row.score:.2f}\n    {row.text}\n")


if __name__ == "__main__":
    main()
