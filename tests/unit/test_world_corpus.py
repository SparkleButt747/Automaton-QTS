"""Tests for qts.world.corpus."""

from __future__ import annotations

from pathlib import Path


def test_corpus_loads_yaml(tmp_path: Path) -> None:  # T-WCP-1
    from qts.world.corpus import PersonaCorpus

    yaml_path = tmp_path / "test_corpus.yaml"
    yaml_path.write_text(
        "powell:\n"
        "  fomc:\n"
        "    hawkish:\n"
        '      - "Inflation remains stubborn."\n'
        '      - "Further rate hikes may be warranted."\n'
        "    dovish:\n"
        '      - "We are seeing progress on inflation."\n'
    )
    corpus = PersonaCorpus.from_yaml(yaml_path)

    assert corpus.lookup("powell", "fomc", "hawkish") == [
        "Inflation remains stubborn.",
        "Further rate hikes may be warranted.",
    ]


def test_corpus_missing_key_returns_empty(tmp_path: Path) -> None:  # T-WCP-2
    from qts.world.corpus import PersonaCorpus

    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("powell:\n  fomc:\n    hawkish:\n      - x\n")
    corpus = PersonaCorpus.from_yaml(yaml_path)

    assert corpus.lookup("trump", "tweet", "rant") == []
    assert corpus.lookup("powell", "missing", "hawkish") == []


def test_corpus_sample_deterministic_with_seed(tmp_path: Path) -> None:  # T-WCP-3
    import random

    from qts.world.corpus import PersonaCorpus

    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text(
        "powell:\n  fomc:\n    hawkish:\n      - a\n      - b\n      - c\n      - d\n"
    )
    corpus = PersonaCorpus.from_yaml(yaml_path)

    s1 = corpus.sample("powell", "fomc", "hawkish", rng=random.Random(42))
    s2 = corpus.sample("powell", "fomc", "hawkish", rng=random.Random(42))
    assert s1 == s2  # same seed -> same sample

    # Different seed may yield same value with 4 items + 1 draw; assert
    # over a sequence instead
    seq_a = [corpus.sample("powell", "fomc", "hawkish", rng=random.Random(42)) for _ in range(5)]
    seq_b = [corpus.sample("powell", "fomc", "hawkish", rng=random.Random(42)) for _ in range(5)]
    assert seq_a == seq_b


def test_corpus_sample_empty_returns_fallback() -> None:  # T-WCP-4
    import random

    from qts.world.corpus import PersonaCorpus

    corpus = PersonaCorpus(entries={})
    out = corpus.sample("nobody", "nothing", "n/a", rng=random.Random(0))
    assert out == ""  # documented fallback


def test_corpus_key_string() -> None:  # T-WCP-5
    from qts.world.corpus import PersonaCorpus

    assert PersonaCorpus.key("powell", "fomc", "hawkish") == "powell:fomc:hawkish"


def test_seed_corpus_loads_from_repo() -> None:  # T-WCP-6
    """The shipped powell_fomc.yaml must load and contain all 3 buckets."""
    from qts.world.corpus import PersonaCorpus

    path = Path("data/world/persona_corpus/powell_fomc.yaml")
    corpus = PersonaCorpus.from_yaml(path)

    for bucket in ("hawkish", "dovish", "neutral"):
        statements = corpus.lookup("powell", "fomc", bucket)
        assert len(statements) >= 3, (
            f"Expected >=3 powell/fomc/{bucket} statements, got {len(statements)}"
        )
