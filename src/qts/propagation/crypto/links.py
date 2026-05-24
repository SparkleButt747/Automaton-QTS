"""LLM filter: candidate token pair -> typed crypto contagion link (spec §5). Mirrors the equity
economic_links.py, with a crypto relation vocabulary and the retry+fallback shipped there."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from qts.oversight.llm_client import LLMClientProtocol

logger = logging.getLogger(__name__)

CryptoRelation = Literal[
    "shares_oracle",
    "bridges_to",
    "collateral_of",
    "integrated_with",
    "entity_exposure",
    "competitor",
    "same_ecosystem",
    "none",
]
# kept (causal) relation types, "none" excluded — index = relation channel for the operator
CRYPTO_RELATIONS: tuple[str, ...] = tuple(r for r in get_args(CryptoRelation) if r != "none")
_VALID: frozenset[str] = frozenset(get_args(CryptoRelation))

CACHE_VERSION = "v1"
CLASSIFY_ATTEMPTS = 3
_NONE_LINK = {"relation": "none", "direction": "none", "confidence": 0.0}
SYSTEM_PROMPT = (
    "You are a crypto-markets risk analyst. Given two crypto assets/protocols, decide whether a "
    "SEVERE adverse event on the first (source) would causally propagate to the second (peer). "
    "Respond ONLY with JSON: "
    '{"relation": one of ["shares_oracle","bridges_to","collateral_of","integrated_with",'
    '"entity_exposure","competitor","same_ecosystem","none"], '
    '"direction": one of ["positive","negative","none"], "confidence": float 0..1}. '
    "Use 'none' when there is no specific contagion channel "
    "(mere co-movement with BTC is NOT a link)."
)


@dataclass(frozen=True)
class CryptoLink:
    source: str
    peer: str
    relation: CryptoRelation
    direction: Literal["positive", "negative", "none"]
    confidence: float


class CryptoLinkClassifier:
    """Classify a candidate (source, peer, context) into a typed contagion link.

    Cached on disk and resilient to transient LLM JSON failures.
    """

    def __init__(self, llm: LLMClientProtocol, cache_dir: Path) -> None:
        self._llm = llm
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, source: str, peer: str, context: str) -> str:
        raw = f"{CACHE_VERSION}|{source}|{peer}|{context}".encode()
        return hashlib.sha256(raw).hexdigest()

    async def classify(self, source: str, peer: str, *, context: str) -> CryptoLink:
        key = self._key(source, peer, context)
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            return self._from_dict(source, peer, json.loads(path.read_text()))
        user = f"Source: {source}\nPeer: {peer}\nContext: {context}"
        raw = await self._query_with_retry(source, peer, user)
        path.write_text(json.dumps(raw))
        return self._from_dict(source, peer, raw)

    async def _query_with_retry(self, source: str, peer: str, user: str) -> dict:
        last: ValueError | None = None
        for _ in range(CLASSIFY_ATTEMPTS):
            try:
                return await self._llm.query_json(SYSTEM_PROMPT, user)
            except ValueError as exc:
                last = exc
        logger.warning(
            "classify(%s->%s) invalid JSON %dx; no-link: %s", source, peer, CLASSIFY_ATTEMPTS, last
        )
        return dict(_NONE_LINK)

    @staticmethod
    def _from_dict(source: str, peer: str, raw: dict) -> CryptoLink:
        relation = str(raw.get("relation", "none")).lower()
        if relation not in _VALID:
            relation = "none"
        direction = str(raw.get("direction", "none")).lower()
        if direction not in ("positive", "negative", "none"):
            direction = "none"
        try:
            conf = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        return CryptoLink(
            source=source,
            peer=peer,
            relation=relation,  # type: ignore[arg-type]
            direction=direction,  # type: ignore[arg-type]
            confidence=conf,
        )
