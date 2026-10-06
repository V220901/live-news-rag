"""Fetch live news from an RSS feed and turn it into documents."""

import feedparser
import pandas as pd


def fetch_documents(rss_url: str) -> pd.DataFrame:
    """Return a DataFrame with columns ``docno`` and ``text``."""
    feed = feedparser.parse(rss_url)
    if not feed.entries:
        raise RuntimeError(f"No entries returned from feed: {rss_url}")

    records = [
        {"docno": str(i), "text": f"{entry.title}. {entry.get('summary', '')}".strip()}
        for i, entry in enumerate(feed.entries)
    ]
    return pd.DataFrame(records)
