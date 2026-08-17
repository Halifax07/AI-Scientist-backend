import asyncio

import httpx

from fsad_scientist.domain.enums import EvidenceStatus
from fsad_scientist.domain.models import EvidenceRecord
from fsad_scientist.evidence.claims import QwenClaimVerifier
from fsad_scientist.evidence.fulltext import FullTextDocument, PdfPage
from fsad_scientist.evidence.search import LiteratureSearchService

ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2405.14529v2</id>
    <published>2024-05-23T12:00:00Z</published>
    <title>AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2</title>
    <summary>Frozen DINOv2 features for few-shot anomaly detection.</summary>
    <author><name>Jane Researcher</name></author>
  </entry>
</feed>
"""


def run(coro):
    return asyncio.run(coro)


def test_search_merges_authoritative_bibliographic_records():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "export.arxiv.org":
            return httpx.Response(200, text=ARXIV_FEED)
        return httpx.Response(
            200,
            json={
                "message": {
                    "items": [
                        {
                            "title": [
                                "AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection "
                                "with DINOv2"
                            ],
                            "DOI": "10.1000/anomalydino",
                            "URL": "https://doi.org/10.1000/anomalydino",
                            "author": [{"given": "Jane", "family": "Researcher"}],
                            "issued": {"date-parts": [[2024, 5, 23]]},
                            "container-title": ["CVPR"],
                            "type": "proceedings-article",
                            "score": 99.0,
                        }
                    ]
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = run(
            LiteratureSearchService(client=client).search(
                "few-shot anomaly detection",
                limit=5,
            )
        )
    finally:
        run(client.aclose())

    assert len(result.records) == 1
    record = result.records[0]
    assert record.arxiv_id == "2405.14529"
    assert record.doi == "10.1000/anomalydino"
    assert record.status == EvidenceStatus.METADATA_VERIFIED
    assert record.verification_scope == "bibliographic"
    assert record.claims == []


def test_identifier_verification_rejects_a_title_mismatch():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "title": ["An unrelated paper"],
                    "DOI": "10.1000/wrong",
                    "issued": {"date-parts": [[2025]]},
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        record = EvidenceRecord(
            title="Expected industrial anomaly detection paper",
            source_type="paper",
            doi="10.1000/wrong",
        )
        verified = run(LiteratureSearchService(client=client).verify(record))
    finally:
        run(client.aclose())

    assert verified.status == EvidenceStatus.REJECTED
    assert verified.verification_scope == "none"


class _FakeClaimClient:
    model = "fake-qwen"

    def __init__(self, *, quote: str) -> None:
        self.quote = quote

    async def complete(self, **_):
        return {
            "verdict": "supports",
            "page_number": 2,
            "quote": self.quote,
            "rationale": "The method statement is explicit.",
        }


def test_claim_verification_requires_a_verbatim_page_anchor():
    record = EvidenceRecord(
        id="evidence_test",
        title="A paper",
        source_type="paper",
        arxiv_id="2405.14529",
        claims=["The method uses frozen DINOv2 patch features."],
        status=EvidenceStatus.METADATA_VERIFIED,
        verification_scope="bibliographic",
    )
    document = FullTextDocument(
        evidence_id=record.id,
        source_url="https://arxiv.org/pdf/2405.14529",
        pdf_path="paper.pdf",
        pdf_sha256="a" * 64,
        pages=[
            PdfPage(page_number=1, text="Introduction"),
            PdfPage(
                page_number=2,
                text="Our method uses frozen DINOv2 patch features for anomaly detection.",
            ),
        ],
        text_digest="b" * 64,
    )
    verifier = QwenClaimVerifier()
    verifier.client = _FakeClaimClient(quote="uses frozen DINOv2 patch features")

    verified = run(verifier.verify(record, document))

    assert verified.status == EvidenceStatus.VERIFIED
    assert verified.verification_scope == "claim"
    assert verified.claim_checks[0].anchored


def test_claim_verification_downgrades_an_unanchored_quote():
    record = EvidenceRecord(
        id="evidence_test",
        title="A paper",
        source_type="paper",
        arxiv_id="2405.14529",
        claims=["A claimed result"],
        status=EvidenceStatus.METADATA_VERIFIED,
        verification_scope="bibliographic",
    )
    document = FullTextDocument(
        evidence_id=record.id,
        source_url="https://arxiv.org/pdf/2405.14529",
        pdf_path="paper.pdf",
        pdf_sha256="a" * 64,
        pages=[PdfPage(page_number=2, text="The actual source passage.")],
        text_digest="b" * 64,
    )
    verifier = QwenClaimVerifier()
    verifier.client = _FakeClaimClient(quote="fabricated quotation")

    verified = run(verifier.verify(record, document))

    assert verified.status == EvidenceStatus.METADATA_VERIFIED
    assert verified.claim_checks[0].verdict == "not_found"
    assert not verified.claim_checks[0].anchored
