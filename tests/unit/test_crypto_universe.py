"""T-CRYPTO-UNIV-*: crypto universe load + node-index/cluster/metadata accessors."""

from __future__ import annotations

from pathlib import Path

from qts.propagation.crypto.universe import load_crypto_universe


def _write_uni(tmp_path: Path) -> Path:
    p = tmp_path / "uni.yaml"
    p.write_text(
        "tokens:\n"
        "  FTT: {cluster: ExchangeToken, is_exchange_token: true, aliases: [ftx token, ftx]}\n"
        "  SOL: {cluster: L1, aliases: [solana]}\n"
        "  UST: {cluster: Stablecoin, is_stablecoin: true, aliases: [terrausd]}\n"
    )
    return p


def test_load_sorts_tokens_and_aligns_metadata(tmp_path: Path) -> None:  # T-CRYPTO-UNIV-1
    uni = load_crypto_universe(_write_uni(tmp_path))
    assert uni.tokens == ("FTT", "SOL", "UST")  # sorted; position = node index
    assert uni.index_of("SOL") == 1
    assert uni.cluster_of("FTT") == "ExchangeToken"
    assert uni.is_exchange_token[uni.index_of("FTT")] is True
    assert uni.is_stablecoin[uni.index_of("UST")] is True
    assert uni.is_stablecoin[uni.index_of("SOL")] is False


def test_alias_map_lowercased_names_only(tmp_path: Path) -> None:  # T-CRYPTO-UNIV-2
    uni = load_crypto_universe(_write_uni(tmp_path))
    amap = uni.alias_map()
    assert amap["solana"] == "SOL"
    assert amap["ftx token"] == "FTT"
    assert "FTT" not in amap  # bare symbols excluded (case-sensitive match handled by extractor)
    assert uni.unique_clusters == ("ExchangeToken", "L1", "Stablecoin")
