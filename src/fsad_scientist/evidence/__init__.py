"""Literature retrieval and evidence verification boundaries."""

from fsad_scientist.evidence.fulltext import ArxivFullTextService, FullTextDocument
from fsad_scientist.evidence.search import (
    EvidenceRetrievalError,
    LiteratureSearchResult,
    LiteratureSearchService,
)

__all__ = [
    "EvidenceRetrievalError",
    "LiteratureSearchResult",
    "LiteratureSearchService",
    "ArxivFullTextService",
    "FullTextDocument",
]
