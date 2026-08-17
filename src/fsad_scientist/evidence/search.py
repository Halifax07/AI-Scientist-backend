from __future__ import annotations

import asyncio
import html
import re
from difflib import SequenceMatcher
from time import monotonic
from typing import Any, Literal
from urllib.parse import quote
from xml.etree import ElementTree

import httpx
from pydantic import BaseModel, Field

from fsad_scientist.domain.enums import EvidenceStatus
from fsad_scientist.domain.models import EvidenceRecord

ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"
ARXIV_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}


class EvidenceRetrievalError(RuntimeError):
    """Raised only when every requested evidence provider fails."""


class LiteratureSearchResult(BaseModel):
    query: str
    records: list[EvidenceRecord] = Field(default_factory=list)
    providers_succeeded: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LiteratureSearchService:
    """Retrieve bibliographic metadata without asking an LLM to invent citations.

    A provider response verifies bibliographic identity only. Paper claims remain
    unverified until a later full-text, passage-level check promotes the record to
    ``EvidenceStatus.VERIFIED`` with ``verification_scope='claim'``.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        mailto: str | None = None,
        timeout_seconds: float = 30.0,
        arxiv_interval_seconds: float = 3.0,
    ) -> None:
        self._client = client
        self.mailto = mailto
        self.timeout_seconds = timeout_seconds
        self.arxiv_interval_seconds = max(0.0, arxiv_interval_seconds)
        self._arxiv_lock = asyncio.Lock()
        self._last_arxiv_request: float | None = None

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        providers: tuple[Literal["arxiv", "crossref"], ...] = ("arxiv", "crossref"),
    ) -> LiteratureSearchResult:
        cleaned_query = " ".join(query.split())
        if not cleaned_query:
            raise ValueError("query cannot be empty")
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")

        operations = {
            "arxiv": self._search_arxiv(cleaned_query, limit=limit),
            "crossref": self._search_crossref(cleaned_query, limit=limit),
        }
        selected = [(name, operations[name]) for name in dict.fromkeys(providers)]
        responses = await asyncio.gather(
            *(operation for _, operation in selected),
            return_exceptions=True,
        )

        records: list[EvidenceRecord] = []
        succeeded: list[str] = []
        warnings: list[str] = []
        for (provider, _), response in zip(selected, responses, strict=True):
            if isinstance(response, BaseException):
                warnings.append(f"{provider}: {type(response).__name__}: {response}")
                continue
            succeeded.append(provider)
            records.extend(response)

        if not succeeded:
            raise EvidenceRetrievalError("; ".join(warnings) or "No provider was selected")

        return LiteratureSearchResult(
            query=cleaned_query,
            records=_deduplicate(records)[:limit],
            providers_succeeded=succeeded,
            warnings=warnings,
        )

    async def verify(self, record: EvidenceRecord) -> EvidenceRecord:
        """Re-fetch a DOI/arXiv record and compare its normalized title."""

        if record.doi:
            candidate = await self._fetch_crossref_doi(record.doi)
            provider = "crossref"
        elif record.arxiv_id:
            candidates = await self._fetch_arxiv_ids([record.arxiv_id])
            candidate = candidates[0] if candidates else None
            provider = "arxiv"
        else:
            raise ValueError("verification requires a DOI or arXiv identifier")

        updated = record.model_copy(deep=True)
        if candidate is None:
            updated.status = EvidenceStatus.REJECTED
            updated.verification_scope = "none"
            updated.verification_notes.append(f"{provider} did not return the identifier")
            return updated

        similarity = _title_similarity(record.title, candidate.title)
        updated.metadata["verification_title_similarity"] = round(similarity, 4)
        updated.metadata["verification_provider"] = provider
        if similarity < 0.86:
            updated.status = EvidenceStatus.REJECTED
            updated.verification_scope = "none"
            updated.verification_notes.append(
                f"Identifier resolved, but title similarity {similarity:.3f} was below 0.86"
            )
            return updated

        merged = _merge_records(updated, candidate)
        merged.status = EvidenceStatus.METADATA_VERIFIED
        merged.verification_scope = "bibliographic"
        merged.verification_notes.append(
            f"Bibliographic identity re-fetched from {provider}; claims are not yet verified"
        )
        return merged

    async def _search_arxiv(self, query: str, *, limit: int) -> list[EvidenceRecord]:
        response = await self._request_arxiv(
            params={
                "search_query": f'all:"{query}"',
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        return _parse_arxiv_feed(response.text)

    async def _fetch_arxiv_ids(self, identifiers: list[str]) -> list[EvidenceRecord]:
        response = await self._request_arxiv(
            params={"id_list": ",".join(_clean_arxiv_id(item) for item in identifiers)},
        )
        return _parse_arxiv_feed(response.text)

    async def _search_crossref(self, query: str, *, limit: int) -> list[EvidenceRecord]:
        params: dict[str, str | int] = {
            "query.bibliographic": query,
            "rows": limit,
            "select": (
                "DOI,title,author,published-print,published-online,issued,URL,"
                "container-title,abstract,type,score"
            ),
        }
        if self.mailto:
            params["mailto"] = self.mailto
        response = await self._request(
            "GET",
            CROSSREF_API,
            params=params,
            headers={"User-Agent": self._crossref_user_agent},
        )
        items = response.json().get("message", {}).get("items", [])
        return [_crossref_record(item) for item in items if item.get("title")]

    async def _fetch_crossref_doi(self, doi: str) -> EvidenceRecord | None:
        params = {"mailto": self.mailto} if self.mailto else None
        response = await self._request(
            "GET",
            f"{CROSSREF_API}/{quote(doi.strip(), safe='')}",
            params=params,
            headers={"User-Agent": self._crossref_user_agent},
            allow_not_found=True,
        )
        if response.status_code == 404:
            return None
        item = response.json().get("message", {})
        return _crossref_record(item) if item.get("title") else None

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        allow_not_found: bool = False,
    ) -> httpx.Response:
        if self._client is not None:
            response = await self._client.request(method, url, params=params, headers=headers)
        else:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=True,
            ) as client:
                response = await client.request(method, url, params=params, headers=headers)
        if allow_not_found and response.status_code == 404:
            return response
        response.raise_for_status()
        return response

    async def _request_arxiv(self, *, params: dict[str, Any]) -> httpx.Response:
        async with self._arxiv_lock:
            if self._last_arxiv_request is not None:
                elapsed = monotonic() - self._last_arxiv_request
                delay = self.arxiv_interval_seconds - elapsed
                if delay > 0:
                    await asyncio.sleep(delay)
            response = await self._request(
                "GET",
                ARXIV_API,
                params=params,
                headers={"User-Agent": "FSAD-Scientist/0.1 (research prototype)"},
            )
            self._last_arxiv_request = monotonic()
            return response

    @property
    def _crossref_user_agent(self) -> str:
        contact = f"; mailto:{self.mailto}" if self.mailto else ""
        return f"FSAD-Scientist/0.1 ({contact or 'research prototype'})"


def _parse_arxiv_feed(xml_text: str) -> list[EvidenceRecord]:
    root = ElementTree.fromstring(xml_text)
    records: list[EvidenceRecord] = []
    for entry in root.findall("atom:entry", ARXIV_NAMESPACE):
        identifier_url = _element_text(entry, "atom:id")
        title = _collapse(_element_text(entry, "atom:title"))
        if not identifier_url or not title:
            continue
        arxiv_id = _clean_arxiv_id(identifier_url.rsplit("/", 1)[-1])
        published = _element_text(entry, "atom:published")
        authors = [
            _collapse(_element_text(author, "atom:name"))
            for author in entry.findall("atom:author", ARXIV_NAMESPACE)
        ]
        doi = _element_text(entry, "{http://arxiv.org/schemas/atom}doi") or None
        records.append(
            EvidenceRecord(
                title=title,
                source_type="paper",
                url=f"https://arxiv.org/abs/{arxiv_id}",
                doi=doi,
                arxiv_id=arxiv_id,
                authors=[item for item in authors if item],
                published_year=_parse_year(published),
                abstract=_collapse(_element_text(entry, "atom:summary")) or None,
                source_provider="arxiv",
                status=EvidenceStatus.METADATA_VERIFIED,
                verification_scope="bibliographic",
                verification_notes=[
                    "Retrieved from the arXiv metadata API; paper claims are unverified"
                ],
                metadata={"published": published},
            )
        )
    return records


def _crossref_record(item: dict[str, Any]) -> EvidenceRecord:
    title = _collapse((item.get("title") or [""])[0])
    authors = [
        _collapse(" ".join(part for part in [author.get("given"), author.get("family")] if part))
        for author in item.get("author", [])
    ]
    venue = _collapse((item.get("container-title") or [""])[0]) or None
    doi = item.get("DOI")
    return EvidenceRecord(
        title=title,
        source_type="paper",
        url=item.get("URL") or (f"https://doi.org/{doi}" if doi else None),
        doi=doi,
        authors=[author for author in authors if author],
        published_year=_crossref_year(item),
        venue=venue,
        abstract=_strip_markup(item.get("abstract")) or None,
        source_provider="crossref",
        status=EvidenceStatus.METADATA_VERIFIED,
        verification_scope="bibliographic",
        verification_notes=[
            "Retrieved from the Crossref REST API; paper claims are unverified"
        ],
        metadata={
            "crossref_type": item.get("type"),
            "crossref_score": item.get("score"),
        },
    )


def _crossref_year(item: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "issued"):
        parts = item.get(key, {}).get("date-parts", [])
        if parts and parts[0]:
            try:
                return int(parts[0][0])
            except (TypeError, ValueError):
                continue
    return None


def _element_text(element: ElementTree.Element, path: str) -> str:
    child = element.find(path, ARXIV_NAMESPACE)
    return child.text.strip() if child is not None and child.text else ""


def _clean_arxiv_id(identifier: str) -> str:
    return re.sub(r"v\d+$", "", identifier.strip()).removeprefix("arXiv:")


def _collapse(value: str) -> str:
    return " ".join(value.split())


def _strip_markup(value: str | None) -> str:
    if not value:
        return ""
    return _collapse(html.unescape(re.sub(r"<[^>]+>", " ", value)))


def _parse_year(value: str) -> int | None:
    match = re.match(r"(\d{4})", value)
    return int(match.group(1)) if match else None


def _normalized_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalized_title(left), _normalized_title(right)).ratio()


def _deduplicate(records: list[EvidenceRecord]) -> list[EvidenceRecord]:
    unique: dict[str, EvidenceRecord] = {}
    aliases: dict[str, str] = {}
    order: list[str] = []
    for record in records:
        keys = [f"title:{_normalized_title(record.title)}"]
        if record.doi:
            keys.append(f"doi:{record.doi.casefold()}")
        if record.arxiv_id:
            keys.append(f"arxiv:{record.arxiv_id.casefold()}")
        existing_key = next((aliases[key] for key in keys if key in aliases), None)
        if existing_key is None:
            primary_key = keys[1] if len(keys) > 1 else keys[0]
            unique[primary_key] = record
            order.append(primary_key)
            for key in keys:
                aliases[key] = primary_key
        else:
            unique[existing_key] = _merge_records(unique[existing_key], record)
            for key in keys:
                aliases[key] = existing_key
    return [unique[key] for key in order]


def _merge_records(primary: EvidenceRecord, secondary: EvidenceRecord) -> EvidenceRecord:
    result = primary.model_copy(deep=True)
    for name in ("url", "doi", "arxiv_id", "published_year", "venue", "abstract"):
        if getattr(result, name) in (None, "") and getattr(secondary, name) not in (None, ""):
            setattr(result, name, getattr(secondary, name))
    result.authors = list(dict.fromkeys([*result.authors, *secondary.authors]))
    result.claims = list(dict.fromkeys([*result.claims, *secondary.claims]))
    result.verification_notes = list(
        dict.fromkeys([*result.verification_notes, *secondary.verification_notes])
    )
    providers = {item for item in [result.source_provider, secondary.source_provider] if item}
    result.source_provider = "+".join(sorted(providers)) or None
    result.metadata.update(secondary.metadata)
    return result
