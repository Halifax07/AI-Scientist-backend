from __future__ import annotations

import math
import re
from dataclasses import dataclass

from fsad_scientist.agents.agentscope_client import AgentScopeJsonClient
from fsad_scientist.domain.enums import EvidenceStatus
from fsad_scientist.domain.models import ClaimVerification, EvidenceRecord
from fsad_scientist.evidence.fulltext import FullTextDocument


@dataclass(frozen=True)
class Passage:
    page_number: int
    text: str


class QwenClaimVerifier:
    """Verify claims against retrieved passages and reject unanchored model output."""

    def __init__(
        self,
        *,
        model: str = "qwen3.7-plus",
        api_key: str | None = None,
    ) -> None:
        self.client = AgentScopeJsonClient(model=model, api_key=api_key)

    async def verify(
        self,
        record: EvidenceRecord,
        document: FullTextDocument,
    ) -> EvidenceRecord:
        if document.evidence_id != record.id:
            raise ValueError("full-text document belongs to another evidence record")
        updated = record.model_copy(deep=True)
        checks: list[ClaimVerification] = []
        for claim in record.claims:
            passages = _rank_passages(claim, document)[:8]
            response = await self.client.complete(
                role_name="CitationClaimVerifier",
                system_prompt=(
                    "你是严格的原文声明核验员。只根据给定页码段落判断 claim。"
                    "supports 表示段落直接支持；contradicts 表示直接冲突；否则 not_found。"
                    "quote 必须逐字复制一个短原文片段并给出正确页码，不允许改写或补全。"
                ),
                payload={
                    "paper_title": record.title,
                    "claim": claim,
                    "passages": [
                        {"page_number": item.page_number, "text": item.text}
                        for item in passages
                    ],
                    "return_schema": {
                        "verdict": "supports|contradicts|not_found",
                        "page_number": "integer|null",
                        "quote": "verbatim string|null",
                        "rationale": "string",
                    },
                },
            )
            checks.append(_validate_check(claim, response, passages, self.client.model))

        updated.claim_checks = checks
        if checks and all(item.anchored for item in checks):
            updated.verification_scope = "claim"
            if any(item.verdict == "contradicts" for item in checks):
                updated.status = EvidenceStatus.REJECTED
            elif all(item.verdict == "supports" for item in checks):
                updated.status = EvidenceStatus.VERIFIED
            else:
                updated.status = EvidenceStatus.METADATA_VERIFIED
        else:
            updated.verification_scope = "bibliographic"
            if updated.status != EvidenceStatus.REJECTED:
                updated.status = EvidenceStatus.METADATA_VERIFIED
        return updated


def _rank_passages(claim: str, document: FullTextDocument) -> list[Passage]:
    claim_tokens = _tokens(claim)
    passages: list[tuple[float, Passage]] = []
    for page in document.pages:
        for text in _chunk(page.text):
            passage_tokens = _tokens(text)
            overlap = len(claim_tokens & passage_tokens)
            score = overlap / math.sqrt(max(1, len(claim_tokens) * len(passage_tokens)))
            passages.append((score, Passage(page.page_number, text)))
    passages.sort(key=lambda item: (item[0], -item[1].page_number), reverse=True)
    return [item for _, item in passages]


def _validate_check(
    claim: str,
    response: dict,
    passages: list[Passage],
    verifier: str,
) -> ClaimVerification:
    verdict = response.get("verdict", "not_found")
    if verdict not in {"supports", "contradicts", "not_found"}:
        verdict = "not_found"
    page_number = response.get("page_number")
    quote = response.get("quote")
    rationale = str(response.get("rationale", "No rationale returned"))
    anchored = False
    if (
        verdict != "not_found"
        and isinstance(page_number, int)
        and isinstance(quote, str)
        and 0 < len(quote) <= 600
    ):
        normalized_quote = _normalize(quote)
        anchored = any(
            item.page_number == page_number and normalized_quote in _normalize(item.text)
            for item in passages
        )
    if not anchored:
        verdict = "not_found"
        page_number = None
        quote = None
        rationale = f"未找到可逐字定位的证据。模型说明：{rationale}"
    return ClaimVerification(
        claim=claim,
        verdict=verdict,
        page_number=page_number,
        quote=quote,
        rationale=rationale,
        anchored=anchored,
        verifier=verifier,
    )


def _chunk(text: str, *, size: int = 1800, overlap: int = 240) -> list[str]:
    collapsed = " ".join(text.split())
    if not collapsed:
        return []
    step = size - overlap
    return [collapsed[start : start + size] for start in range(0, len(collapsed), step)]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.casefold()))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()
