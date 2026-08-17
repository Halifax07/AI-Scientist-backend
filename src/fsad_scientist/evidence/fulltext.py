from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from fsad_scientist.domain.models import EvidenceRecord, utc_now


class PdfPage(BaseModel):
    page_number: int = Field(ge=1)
    text: str


class FullTextDocument(BaseModel):
    schema_version: str = "1.0"
    evidence_id: str
    source_url: str
    pdf_path: str
    manifest_path: str | None = None
    pdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pages: list[PdfPage]
    text_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    extracted_at: datetime = Field(default_factory=utc_now)

    @property
    def page_count(self) -> int:
        return len(self.pages)


class ArxivFullTextService:
    """Download an identified arXiv PDF and extract page-addressable text."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        client: httpx.AsyncClient | None = None,
        maximum_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        self.artifact_root = artifact_root.expanduser().resolve()
        self.client = client
        self.maximum_bytes = maximum_bytes

    async def fetch_and_extract(
        self,
        record: EvidenceRecord,
        *,
        force: bool = False,
    ) -> FullTextDocument:
        if not record.arxiv_id:
            raise ValueError("automatic full-text retrieval currently requires an arXiv id")
        arxiv_id = _clean_arxiv_id(record.arxiv_id)
        source_url = f"https://arxiv.org/pdf/{arxiv_id}"
        destination = self.artifact_root / "evidence" / "pdfs" / f"{arxiv_id}.pdf"
        if force or not destination.is_file():
            content = await self._download(source_url)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".pdf.tmp")
            temporary.write_bytes(content)
            temporary.replace(destination)

        output = self.artifact_root / "evidence" / "fulltext" / f"{arxiv_id}.json"
        if output.is_file() and not force:
            return FullTextDocument.model_validate_json(output.read_text(encoding="utf-8"))
        document = await _extract_pdf(record.id, source_url, destination)
        document.manifest_path = str(output.resolve())
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_suffix(".json.tmp")
        temporary_output.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        temporary_output.replace(output)
        return document

    async def _download(self, url: str) -> bytes:
        if self.client is not None:
            response = await self.client.get(url, follow_redirects=True)
        else:
            async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": "FSAD-Scientist/0.1 (research prototype)"},
                )
        response.raise_for_status()
        content = response.content
        if len(content) > self.maximum_bytes:
            raise ValueError(f"PDF exceeds {self.maximum_bytes} bytes")
        if not content.startswith(b"%PDF"):
            raise ValueError("arXiv response is not a PDF")
        return content


async def _extract_pdf(
    evidence_id: str,
    source_url: str,
    pdf_path: Path,
) -> FullTextDocument:
    import asyncio

    return await asyncio.to_thread(_extract_pdf_sync, evidence_id, source_url, pdf_path)


def _extract_pdf_sync(
    evidence_id: str,
    source_url: str,
    pdf_path: Path,
) -> FullTextDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF extraction requires pypdf") from exc

    pdf_bytes = pdf_path.read_bytes()
    reader = PdfReader(pdf_path)
    pages = [
        PdfPage(page_number=index, text=page.extract_text() or "")
        for index, page in enumerate(reader.pages, start=1)
    ]
    text_payload: dict[str, Any] = {
        "evidence_id": evidence_id,
        "pages": [page.model_dump(mode="json") for page in pages],
    }
    return FullTextDocument(
        evidence_id=evidence_id,
        source_url=source_url,
        pdf_path=str(pdf_path.resolve()),
        pdf_sha256=hashlib.sha256(pdf_bytes).hexdigest(),
        pages=pages,
        text_digest=hashlib.sha256(
            json.dumps(text_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    )


def _clean_arxiv_id(value: str) -> str:
    cleaned = re.sub(r"v\d+$", "", value.strip()).removeprefix("arXiv:")
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})", cleaned, re.IGNORECASE):
        raise ValueError(f"invalid arXiv identifier: {value}")
    return cleaned
