"""Hand-validate the news classifier on the curated dataset.

Walks every TextEvent in data/real/fomc/<DATE>, classifies each via
the Qwen-backed NewsClassifier (llama.cpp by default), and prints the
structured output for visual inspection. Use this BEFORE running the v2
acceptance test to confirm the model is reading the text sensibly — and
to warm the cache so the acceptance test runs without a live LLM.

Usage:
    # default: llama.cpp server on http://localhost:8080
    .venv/bin/python -m scripts.validate_news_classifier data/real/fomc/2023-12-13

    # point at a non-default llama-server, or switch backend
    .venv/bin/python -m scripts.validate_news_classifier data/real/fomc/2023-12-13 \\
        --base-url http://localhost:9090 --model qwen2.5-7b-instruct
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from qts.data.real_episode import RealEpisode
from qts.macro.news_classifier import NewsClassifier
from qts.oversight.llm_client import create_llm_client

logger = logging.getLogger(__name__)


async def _main(root: Path, cache_dir: Path, llm_kwargs: dict) -> int:
    episode = RealEpisode.from_disk(root, symbol="BTCUSDT", source=f"validate:{root.name}")
    llm = create_llm_client(**llm_kwargs)
    classifier = NewsClassifier(llm_client=llm, cache_dir=cache_dir)

    print(f"\nValidating {len(episode.text_events)} text events from {root}:")
    print("-" * 80)
    for i, event in enumerate(episode.text_events, 1):
        signal = await classifier.classify_async(event)
        print(
            f"[{i:>2}] src={event.source:>20s} "
            f"dir={signal.direction:>7s} "
            f"conf={signal.confidence:.2f} rel={signal.relevance:.2f} "
            f"mag={signal.magnitude:.2f} | "
            f"alpha={signal.alpha_contribution():+.3f}"
        )
        preview = event.text[:100].replace("\n", " ")
        print(f"     text: {preview!r}...")
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hand-validate news classifier on curated data.")
    parser.add_argument("root", help="Curated dataset directory")
    parser.add_argument(
        "--cache-dir",
        default="data/news_cache",
        help="Where to cache classifier responses (default: data/news_cache)",
    )
    parser.add_argument(
        "--backend",
        default="llamacpp",
        choices=["llamacpp", "ollama", "anthropic"],
        help="LLM backend (default: llamacpp)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the server base URL (e.g. http://localhost:8080 for llama.cpp)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name/alias",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    llm_kwargs: dict = {"backend": args.backend}
    if args.base_url is not None:
        llm_kwargs["base_url"] = args.base_url
    if args.model is not None:
        llm_kwargs["model"] = args.model

    return asyncio.run(_main(Path(args.root), Path(args.cache_dir), llm_kwargs))


if __name__ == "__main__":
    sys.exit(main())
