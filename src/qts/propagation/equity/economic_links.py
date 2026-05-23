"""LLM filter: co-mention -> typed causal economic link (Path A v2 Phase 1, spec §5.2)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from qts.oversight.llm_client import LLMClientProtocol

Relation = Literal["competitor", "supplier", "customer", "partner", "none"]
_VALID: frozenset[str] = frozenset(get_args(Relation))

CACHE_VERSION = "v1"
SYSTEM_PROMPT = (
    "You are a financial-relations analyst. Given two US-listed companies co-mentioned in news, "
    "decide whether there is a DIRECT economic link from the first (source) to the second (peer) "
    "such that material news about the source would causally move the peer. "
    "Respond ONLY with JSON: "
    '{"relation": one of ["competitor","supplier","customer","partner","none"], '
    '"direction": one of ["positive","negative","none"], "confidence": float 0..1}. '
    "Use 'none' when the co-mention is incidental (e.g. both in a market wrap) with no causal link."
)


@dataclass(frozen=True)
class EconomicLink:
    source: str
    peer: str
    relation: Relation
    direction: Literal["positive", "negative", "none"]
    confidence: float


class EconomicLinkClassifier:
    """Classify a co-mention (source, peer, context) into a typed link; content-addressed cache."""

    def __init__(self, llm: LLMClientProtocol, cache_dir: Path) -> None:
        self._llm = llm
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, source: str, peer: str, context: str) -> str:
        raw = f"{CACHE_VERSION}|{source}|{peer}|{context}".encode()
        return hashlib.sha256(raw).hexdigest()

    async def classify(self, source: str, peer: str, *, context: str) -> EconomicLink:
        key = self._key(source, peer, context)
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            return self._from_dict(source, peer, json.loads(path.read_text()))
        user = f"Source: {source}\nPeer: {peer}\nNews context: {context}"
        raw = await self._llm.query_json(SYSTEM_PROMPT, user)
        path.write_text(json.dumps(raw))
        return self._from_dict(source, peer, raw)

    @staticmethod
    def _from_dict(source: str, peer: str, raw: dict) -> EconomicLink:
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
        return EconomicLink(
            source=source,
            peer=peer,
            relation=relation,  # type: ignore[arg-type]
            direction=direction,  # type: ignore[arg-type]
            confidence=conf,
        )
