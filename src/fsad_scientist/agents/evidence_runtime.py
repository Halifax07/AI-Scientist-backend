from __future__ import annotations

from fsad_scientist.agents.contracts import ScientistRuntime
from fsad_scientist.domain.enums import EvidenceStatus
from fsad_scientist.domain.models import ArtifactRecord, EvidenceRecord, ResearchProject
from fsad_scientist.evidence.search import LiteratureSearchService


class EvidenceEnabledRuntime:
    """Decorate a cognitive runtime with live, tool-verified literature retrieval."""

    def __init__(
        self,
        delegate: ScientistRuntime,
        service: LiteratureSearchService,
        *,
        results_per_query: int = 4,
        max_queries: int = 4,
    ) -> None:
        self.delegate = delegate
        self.service = service
        self.results_per_query = results_per_query
        self.max_queries = max_queries
        self.name = f"{delegate.name}+live-evidence"

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)

    async def gather_evidence(
        self,
        project: ResearchProject,
    ) -> tuple[list[EvidenceRecord], ArtifactRecord]:
        seeded, search_plan = await self.delegate.gather_evidence(project)
        queries = list(search_plan.payload.get("queries", []))[: self.max_queries]
        records = list(seeded)
        retrievals: list[dict[str, object]] = []
        for query in queries:
            try:
                result = await self.service.search(
                    str(query),
                    limit=self.results_per_query,
                )
            except Exception as exc:  # a partial provider outage must not invent evidence
                retrievals.append(
                    {
                        "query": str(query),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            records.extend(result.records)
            retrievals.append(
                {
                    "query": result.query,
                    "status": "completed",
                    "providers": result.providers_succeeded,
                    "warnings": result.warnings,
                    "record_count": len(result.records),
                }
            )

        records = _merge_records(records)
        bibliographic_count = sum(
            item.status == EvidenceStatus.METADATA_VERIFIED for item in records
        )
        artifact = ArtifactRecord(
            kind="evidence_search_report",
            title="真实文献检索与书目身份校验报告",
            verified=bibliographic_count > 0,
            provenance=["arxiv_metadata_api", "crossref_rest_api", self.name],
            payload={
                "queries": retrievals,
                "record_count": len(records),
                "bibliographic_verified_count": bibliographic_count,
                "claim_verified_count": sum(
                    item.verification_scope == "claim" for item in records
                ),
                "integrity_note": (
                    "metadata_verified 只代表题名/作者/标识符来自权威元数据接口；"
                    "不得据此把摘要中的方法或效果声明标记为已验证。"
                ),
            },
        )
        return records, artifact


def _merge_records(records: list[EvidenceRecord]) -> list[EvidenceRecord]:
    merged: dict[str, EvidenceRecord] = {}
    order: list[str] = []
    title_aliases: dict[str, str] = {}
    for record in records:
        title = " ".join(record.title.casefold().split())
        preferred = (
            f"doi:{record.doi.casefold()}"
            if record.doi
            else f"arxiv:{record.arxiv_id.casefold()}"
            if record.arxiv_id
            else f"title:{title}"
        )
        key = title_aliases.get(title, preferred)
        if key not in merged:
            merged[key] = record
            order.append(key)
        else:
            current = merged[key].model_copy(deep=True)
            incoming = record
            for field in (
                "url",
                "doi",
                "arxiv_id",
                "published_year",
                "venue",
                "abstract",
                "source_provider",
            ):
                if getattr(current, field) in (None, "") and getattr(incoming, field) not in (
                    None,
                    "",
                ):
                    setattr(current, field, getattr(incoming, field))
            current.authors = list(dict.fromkeys([*current.authors, *incoming.authors]))
            current.claims = list(dict.fromkeys([*current.claims, *incoming.claims]))
            if incoming.status == EvidenceStatus.METADATA_VERIFIED:
                current.status = EvidenceStatus.METADATA_VERIFIED
                current.verification_scope = "bibliographic"
            current.verification_notes = list(
                dict.fromkeys(
                    [*current.verification_notes, *incoming.verification_notes]
                )
            )
            current.metadata.update(incoming.metadata)
            merged[key] = current
        title_aliases[title] = key
    return [merged[key] for key in order]
