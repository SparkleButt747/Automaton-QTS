"""Reddit social sentiment provider."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ── Domain types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SocialPost:
    """A single Reddit post or comment relevant to a given symbol.

    Attributes:
        text: Post title (and selftext if available).
        score: Reddit upvote score (net upvotes).
        upvote_ratio: Fraction of upvotes (0.0 – 1.0).
        timestamp: UTC datetime the post was created.
        subreddit: Subreddit name (without ``r/`` prefix).
        url: Permalink to the post.
    """

    text: str
    score: int
    upvote_ratio: float
    timestamp: datetime
    subreddit: str
    url: str


# ── Protocol (for dependency injection) ──────────────────────────────────────


@runtime_checkable
class RedditProtocol(Protocol):
    """Dependency-injection interface for Reddit sentiment providers."""

    async def fetch_mentions(
        self,
        symbol: str,
        subreddits: list[str],
        limit: int = 100,
    ) -> list[SocialPost]:
        """Fetch Reddit posts mentioning *symbol*.

        Args:
            symbol: Ticker symbol to search for (e.g. ``"TSLA"``).
            subreddits: List of subreddit names to search.
            limit: Maximum posts to return across all subreddits (default 100).

        Returns:
            List of :class:`SocialPost` objects ordered newest-first.
        """
        ...


# ── Implementation ────────────────────────────────────────────────────────────


class RedditSentimentProvider:
    """Async Reddit sentiment provider backed by the PRAW library.

    Searches the provided subreddits for posts mentioning *symbol* and returns
    them as :class:`SocialPost` objects for downstream sentiment analysis.

    The ``praw`` package is imported lazily; if it is absent, :meth:`fetch_mentions`
    raises ``ImportError`` with a helpful message.

    Args:
        client_id: Reddit OAuth2 client ID.
        client_secret: Reddit OAuth2 client secret.
        user_agent: User-agent string for the Reddit API (default ``"QTS/1.0"``).
    """

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        user_agent: str = "QTS/1.0",
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._reddit: Any = None

    def _get_reddit(self) -> Any:
        """Lazily initialise and return the PRAW Reddit client.

        Returns:
            Configured ``praw.Reddit`` instance.

        Raises:
            ImportError: If ``praw`` is not installed.
        """
        if self._reddit is None:
            try:
                import praw
            except ImportError as exc:
                raise ImportError(
                    "RedditSentimentProvider requires 'praw'. "
                    "Install with: pip install praw"
                ) from exc

            self._reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
        return self._reddit

    def _build_query(self, symbol: str) -> str:
        """Build a Reddit search query for the given symbol.

        Args:
            symbol: Ticker symbol (e.g. ``"TSLA"``).

        Returns:
            Search query string with common ticker notations.
        """
        # Match ``TSLA``, ``$TSLA``, and ``#TSLA``
        return f"{symbol} OR ${symbol}"

    def _post_to_social(self, submission: Any, subreddit: str) -> SocialPost:
        """Convert a PRAW Submission object to a :class:`SocialPost`.

        Args:
            submission: PRAW ``Submission`` object.
            subreddit: Subreddit name for this submission.

        Returns:
            Populated :class:`SocialPost`.
        """
        body = submission.title
        if hasattr(submission, "selftext") and submission.selftext:
            body = f"{submission.title} {submission.selftext}"

        timestamp = datetime.fromtimestamp(
            float(submission.created_utc), tz=timezone.utc
        )

        return SocialPost(
            text=body,
            score=int(submission.score),
            upvote_ratio=float(getattr(submission, "upvote_ratio", 0.5)),
            timestamp=timestamp,
            subreddit=subreddit,
            url=f"https://reddit.com{submission.permalink}",
        )

    async def fetch_mentions(
        self,
        symbol: str,
        subreddits: list[str],
        limit: int = 100,
    ) -> list[SocialPost]:
        """Fetch Reddit posts mentioning *symbol* across *subreddits*.

        Note: PRAW is a synchronous library. This method runs PRAW calls
        in a thread executor to remain non-blocking in an async context.

        Args:
            symbol: Ticker symbol to search for.
            subreddits: Subreddit names to search (without ``r/`` prefix).
            limit: Maximum number of posts to return (default 100).

        Returns:
            List of :class:`SocialPost` objects ordered newest-first.

        Raises:
            ImportError: If ``praw`` is not installed.
        """
        import asyncio

        reddit = self._get_reddit()
        per_sub_limit = max(1, limit // max(len(subreddits), 1))

        posts: list[SocialPost] = []
        query = self._build_query(symbol)

        def _fetch_sync() -> list[SocialPost]:
            collected: list[SocialPost] = []
            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)
                    for submission in subreddit.search(
                        query, sort="new", time_filter="week", limit=per_sub_limit
                    ):
                        collected.append(self._post_to_social(submission, sub_name))
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "RedditSentimentProvider: error fetching r/%s for %s",
                        sub_name,
                        symbol,
                        exc_info=True,
                    )
            return collected

        loop = asyncio.get_event_loop()
        posts = await loop.run_in_executor(None, _fetch_sync)
        # Sort newest-first and respect global limit
        posts.sort(key=lambda p: p.timestamp, reverse=True)
        return posts[:limit]
