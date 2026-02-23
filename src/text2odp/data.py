from __future__ import annotations

from dataclasses import dataclass

import requests

from .schemas import PaperRecord


@dataclass
class SemanticScholarCollector:
    endpoint: str = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(self, query: str, limit: int = 20) -> list[PaperRecord]:
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,year,venue",
        }
        import requests

        response = requests.get(self.endpoint, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()

        papers: list[PaperRecord] = []
        for item in payload.get("data", []):
            abstract = item.get("abstract")
            title = item.get("title")
            if not abstract or not title:
                continue
            papers.append(
                PaperRecord(
                    paper_id=item.get("paperId", title),
                    title=title,
                    abstract=abstract,
                    venue=item.get("venue"),
                    year=item.get("year"),
                    source="semantic_scholar",
                )
            )
        return papers
