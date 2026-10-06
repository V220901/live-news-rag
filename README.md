# Live-News RAG

A retrieval-augmented generation (RAG) pipeline that answers questions about **today's** stock news, something a standalone LLM can't do because its knowledge is frozen at training time.

It pulls a live Yahoo Finance RSS feed, indexes it with **PyTerrier**, retrieves the most relevant articles with **BM25** (sparse retrieval), and feeds them to **Llama-3.2-3B-Instruct** so the answer is grounded in the retrieved evidence.

## How it works

```
RSS feed ──> Documents ──> PyTerrier index ──> BM25 top-k ──> Augmented prompt ──> Llama 3.2 3B ──> Grounded answer
```

| Stage | Module | What it does |
|---|---|---|
| Fetch | `src/live_rag/data.py` | Parses the RSS feed into `docno` / `text` records |
| Index | `src/live_rag/retrieval.py` | Builds a Terrier inverted index |
| Retrieve | `src/live_rag/retrieval.py` | BM25 ranking, returns top-k articles with scores |
| Augment | `src/live_rag/pipeline.py` | Injects retrieved context into a strict "answer only from context" prompt |
| Generate | `src/live_rag/generation.py` | Runs the LLM via Hugging Face `transformers` |

The CLI answers each question twice, **without** and **with** RAG, and prints the evidence used.

## Project structure

```
live-news-rag/
├── main.py                 # CLI entry point
├── requirements.txt
├── src/live_rag/
│   ├── config.py           # ticker, model, top-k, prompts
│   ├── data.py             # RSS ingestion
│   ├── retrieval.py        # PyTerrier indexing + BM25
│   ├── generation.py       # LLM wrapper
│   └── pipeline.py         # end-to-end orchestration
└── docs/
    └── sample_output.md    # example run
```

## Run it

**Requirements:** Python 3.10+, Java (PyTerrier needs a JDK), and a **CUDA GPU** for the LLM. No local GPU? Use Google Colab's free T4:

1. Open a new Colab notebook and set **Runtime → Change runtime type → T4 GPU**.
2. Run:

```python
!git clone https://github.com/V220901/live-news-rag.git
%cd live-news-rag
!pip install -q -r requirements.txt
!python main.py "What is driving Tesla's stock price this week?"
```

Other tickers and options:

```bash
python main.py --ticker NVDA --top-k 5 "Why is Nvidia moving today?"
```

## Example

Question: *What specific news events or reports are driving Tesla's (TSLA) stock price this week?*

- **Without RAG:** the model says it has no real-time access and points to news websites.
- **With RAG:** it cites the Q3 2026 delivery beat, the stabilizing EV demand comment, and Dan Ives' 2027 outlook, all taken from the retrieved articles.

See [`docs/sample_output.md`](docs/sample_output.md) for the full output.

## Notes and limitations

- Results change with the news cycle because the feed is live.
- BM25 is lexical, so it can miss paraphrases. A natural next step is dense or hybrid retrieval (e.g. embeddings + BM25 with rank fusion).
- The corpus is only about 16 headline-and-summary snippets. Fetching full article text would give richer context.
- The strict prompt reduces hallucination but does not eliminate it. Adding citation markers and an evaluation set would be the next improvement.

## Tech stack

Python · PyTerrier (BM25) · feedparser · Hugging Face Transformers · PyTorch · Llama-3.2-3B-Instruct
