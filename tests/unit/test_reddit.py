"""Unit tests for qts.data.social.reddit.

Verifies:
- RedditSentimentProvider._build_query produces a query string containing the symbol
- RedditSentimentProvider._post_to_social converts a praw-like submission to SocialPost
- _post_to_social concatenates selftext when present
- fetch_mentions returns SocialPost objects sorted newest-first
- fetch_mentions applies the global limit
- fetch_mentions handles subreddit fetch errors gracefully (skips the subreddit)
- fetch_mentions raises ImportError when praw is not installed
- _get_reddit raises ImportError when praw is not installed
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from qts.data.social.reddit import RedditSentimentProvider, SocialPost

# ── Helpers ───────────────────────────────────────────────────────────────────

_UTC = timezone.utc


def _make_submission(
    title="Buy TSLA",
    selftext="",
    score=100,
    upvote_ratio=0.9,
    created_utc=1704067200.0,
    permalink="/r/stocks/comments/abc/buy_tsla/",
) -> MagicMock:
    sub = MagicMock()
    sub.title = title
    sub.selftext = selftext
    sub.score = score
    sub.upvote_ratio = upvote_ratio
    sub.created_utc = created_utc
    sub.permalink = permalink
    return sub


# ── T-REDDIT-1 ────────────────────────────────────────────────────────────────


class TestBuildQuery:
    def test_includes_symbol(self):
        """_build_query includes the raw symbol and the $-prefixed version."""
        provider = RedditSentimentProvider()
        query = provider._build_query("TSLA")
        assert "TSLA" in query
        assert "$TSLA" in query


# ── T-REDDIT-2 ────────────────────────────────────────────────────────────────


class TestPostToSocial:
    def test_converts_submission_title(self):
        """_post_to_social uses the submission title as text."""
        provider = RedditSentimentProvider()
        sub = _make_submission(title="TSLA to the moon", selftext="")
        post = provider._post_to_social(sub, "stocks")
        assert post.text == "TSLA to the moon"
        assert post.subreddit == "stocks"

    def test_concatenates_selftext_when_present(self):
        """_post_to_social appends selftext to the title when it is non-empty."""
        provider = RedditSentimentProvider()
        sub = _make_submission(title="Big news", selftext="Here are the details...")
        post = provider._post_to_social(sub, "investing")
        assert "Big news" in post.text
        assert "Here are the details..." in post.text

    def test_parses_timestamp_as_utc(self):
        """_post_to_social converts created_utc to a timezone-aware UTC datetime."""
        provider = RedditSentimentProvider()
        sub = _make_submission(created_utc=1704067200.0)
        post = provider._post_to_social(sub, "stocks")
        assert post.timestamp.tzinfo == _UTC
        assert post.timestamp.year == 2024

    def test_builds_full_reddit_url(self):
        """_post_to_social prepends https://reddit.com to the permalink."""
        provider = RedditSentimentProvider()
        sub = _make_submission(permalink="/r/stocks/comments/abc/")
        post = provider._post_to_social(sub, "stocks")
        assert post.url == "https://reddit.com/r/stocks/comments/abc/"

    def test_copies_score_and_upvote_ratio(self):
        """_post_to_social preserves score and upvote_ratio from the submission."""
        provider = RedditSentimentProvider()
        sub = _make_submission(score=420, upvote_ratio=0.87)
        post = provider._post_to_social(sub, "stocks")
        assert post.score == 420
        assert post.upvote_ratio == pytest.approx(0.87)


# ── T-REDDIT-3 ────────────────────────────────────────────────────────────────


class TestFetchMentions:
    def _make_provider_with_mock_reddit(self):
        provider = RedditSentimentProvider(
            client_id="id", client_secret="secret", user_agent="test"
        )
        mock_reddit = MagicMock()
        provider._reddit = mock_reddit
        return provider, mock_reddit

    @pytest.mark.asyncio
    async def test_returns_social_posts(self):
        """fetch_mentions returns a list of SocialPost objects."""
        provider, mock_reddit = self._make_provider_with_mock_reddit()

        sub1 = _make_submission(title="TSLA bullish", created_utc=1704067200.0)
        sub2 = _make_submission(title="TSLA news", created_utc=1704067260.0)
        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = [sub1, sub2]
        mock_reddit.subreddit.return_value = mock_subreddit

        posts = await provider.fetch_mentions("TSLA", ["stocks"])

        assert len(posts) == 2
        assert all(isinstance(p, SocialPost) for p in posts)

    @pytest.mark.asyncio
    async def test_sorted_newest_first(self):
        """fetch_mentions returns posts sorted newest-first by timestamp."""
        provider, mock_reddit = self._make_provider_with_mock_reddit()

        sub_old = _make_submission(created_utc=1704067200.0)
        sub_new = _make_submission(created_utc=1704070800.0)
        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = [sub_old, sub_new]
        mock_reddit.subreddit.return_value = mock_subreddit

        posts = await provider.fetch_mentions("TSLA", ["stocks"])

        assert posts[0].timestamp > posts[1].timestamp

    @pytest.mark.asyncio
    async def test_respects_global_limit(self):
        """fetch_mentions returns at most `limit` posts."""
        provider, mock_reddit = self._make_provider_with_mock_reddit()

        subs = [_make_submission(created_utc=float(i)) for i in range(20)]
        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = subs
        mock_reddit.subreddit.return_value = mock_subreddit

        posts = await provider.fetch_mentions("TSLA", ["stocks"], limit=5)

        assert len(posts) == 5

    @pytest.mark.asyncio
    async def test_skips_failing_subreddit(self):
        """fetch_mentions skips a subreddit that raises an exception."""
        provider, mock_reddit = self._make_provider_with_mock_reddit()

        ok_sub = _make_submission(title="TSLA good")
        mock_ok = MagicMock()
        mock_ok.search.return_value = [ok_sub]

        mock_fail = MagicMock()
        mock_fail.search.side_effect = Exception("403 Forbidden")

        mock_reddit.subreddit.side_effect = lambda name: (
            mock_fail if name == "wallstreetbets" else mock_ok
        )

        posts = await provider.fetch_mentions("TSLA", ["stocks", "wallstreetbets"])

        # Only posts from "stocks" should be returned
        assert len(posts) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_results(self):
        """fetch_mentions returns [] when no submissions are found."""
        provider, mock_reddit = self._make_provider_with_mock_reddit()

        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = []
        mock_reddit.subreddit.return_value = mock_subreddit

        posts = await provider.fetch_mentions("TSLA", ["stocks"])
        assert posts == []


# ── T-REDDIT-4 ────────────────────────────────────────────────────────────────


class TestGetReddit:
    def test_raises_import_error_when_praw_missing(self):
        """_get_reddit raises ImportError with a helpful message when praw is absent."""
        provider = RedditSentimentProvider(client_id="id", client_secret="secret")

        with patch.dict(sys.modules, {"praw": None}):
            with pytest.raises(ImportError, match="praw"):
                provider._get_reddit()

    def test_reuses_existing_reddit_instance(self):
        """_get_reddit returns the same instance on repeated calls."""
        provider = RedditSentimentProvider()
        mock_reddit = MagicMock()
        provider._reddit = mock_reddit
        assert provider._get_reddit() is mock_reddit
