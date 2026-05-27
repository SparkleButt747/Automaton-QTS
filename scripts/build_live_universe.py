"""Build the live contagion universe + structural seed as the union of all per-event
configs — the live strategy watches everywhere contagion has been observed."""

from pathlib import Path

from qts.propagation.crypto.structural_links import build_live_structural_links
from qts.propagation.crypto.universe import build_live_universe

if __name__ == "__main__":
    uni = build_live_universe(
        Path("config/universe"), Path("config/universe/crypto_contagion_live.yaml")
    )
    links = build_live_structural_links(
        Path("config/links"), Path("config/links/crypto_structural_live.yaml")
    )
    print(f"universe: {len(uni)} tokens -> config/universe/crypto_contagion_live.yaml")
    print(f"structural: {len(links)} links -> config/links/crypto_structural_live.yaml")
