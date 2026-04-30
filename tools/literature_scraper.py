#!/usr/bin/env python3
"""
Literature scraper for the ICPA cotton boll semantic 3D reconstruction paper.

The script queries public scholarly APIs and writes:
  - literature_results.csv
  - literature_results.md

It intentionally uses only the Python standard library so it can run on a
fresh machine. Queries are grouped around the paper's defensible novelty:
semantic correspondence for 3D crop phenotyping, defoliation visibility,
cotton boll morphology, 3DGS/NeRF/SfM, SAM/DINOv2, and agriculture LLMs.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


OPENALEX_ENDPOINT = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"


QUERIES = {
    "cotton_3d_phenotyping": [
        "cotton boll 3D reconstruction UAV",
        "cotton boll phenotyping point cloud UAV",
        "cotton boll Gaussian Splatting phenotyping",
        "cotton boll volume estimation 3D",
    ],
    "defoliation_visibility": [
        "cotton boll counting before after defoliation UAV",
        "cotton defoliation boll opening rate UAV deep learning",
        "cotton boll extraction defoliation RGB imagery",
    ],
    "semantic_reconstruction": [
        "semantic feature fields 3D reconstruction DINOv2",
        "DINOv2 dense correspondence 3D reconstruction",
        "ICLR 2025 DINOv2 correspondence dense features",
        "NeurIPS 2024 foundation model dense matching 3D reconstruction",
        "semantic bundle adjustment feature matching",
        "foundation model features multi view stereo",
    ],
    "foundation_3d": [
        "DUSt3R 3D reconstruction CVPR 2024",
        "MASt3R image matching 3D ECCV 2024",
        "Feature 3DGS semantic feature field CVPR 2024",
        "COLMAP free 3D Gaussian Splatting CVPR 2024",
    ],
    "segmentation": [
        "SAM 2 agriculture segmentation cotton boll",
        "Segment Anything agricultural image segmentation crop",
        "YOLO cotton boll detection UAV",
    ],
    "agriculture_llm": [
        "large language models agriculture decision support",
        "AgriLLM farmer queries",
        "AgroGPT agricultural vision language model",
        "AgriVLM agricultural visual language large model",
    ],
}


TOPIC_TERMS = {
    "cotton_3d_phenotyping": ["cotton", "boll", "3d", "uav", "phenotyping", "point cloud", "gaussian"],
    "defoliation_visibility": ["cotton", "boll", "defoliation", "uav", "opening", "rgb"],
    "semantic_reconstruction": ["semantic", "feature", "dinov2", "correspondence", "matching", "3d", "reconstruction"],
    "foundation_3d": ["dust3r", "mast3r", "3dgs", "gaussian", "splatting", "cvpr", "eccv", "reconstruction"],
    "segmentation": ["sam", "segment anything", "agriculture", "crop", "cotton", "segmentation", "yolo"],
    "agriculture_llm": ["agri", "agriculture", "farmer", "crop", "language model", "vision-language", "llm"],
}


@dataclass(frozen=True)
class Paper:
    source: str
    topic: str
    query: str
    title: str
    year: str
    venue: str
    authors: str
    doi: str
    url: str
    abstract: str
    citations: str


def request_json(url: str, headers: dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def inverted_index_to_abstract(index: dict | None) -> str:
    if not index:
        return ""
    positions = []
    for word, offsets in index.items():
        for offset in offsets:
            positions.append((offset, word))
    return " ".join(word for _, word in sorted(positions))


def query_openalex(topic: str, query: str, limit: int) -> list[Paper]:
    params = {
        "search": query,
        "per-page": str(limit),
        "sort": "cited_by_count:desc",
    }
    url = f"{OPENALEX_ENDPOINT}?{urllib.parse.urlencode(params)}"
    data = request_json(url)
    papers = []
    for work in data.get("results", []):
        authorships = work.get("authorships", [])[:5]
        authors = ", ".join(
            clean_text(a.get("author", {}).get("display_name")) for a in authorships
        )
        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        venue = source.get("display_name", "")
        papers.append(
            Paper(
                source="OpenAlex",
                topic=topic,
                query=query,
                title=clean_text(work.get("title")),
                year=str(work.get("publication_year") or ""),
                venue=clean_text(venue),
                authors=authors,
                doi=clean_text(work.get("doi")),
                url=clean_text(work.get("id")),
                abstract=clean_text(inverted_index_to_abstract(work.get("abstract_inverted_index"))),
                citations=str(work.get("cited_by_count") or ""),
            )
        )
    return papers


def query_semantic_scholar(topic: str, query: str, limit: int) -> list[Paper]:
    params = {
        "query": query,
        "limit": str(limit),
        "fields": "title,year,venue,authors,abstract,citationCount,url,externalIds",
    }
    url = f"{SEMANTIC_SCHOLAR_ENDPOINT}?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": "ICPA-literature-scout/1.0"}
    data = request_json(url, headers=headers)
    papers = []
    for work in data.get("data", []):
        authors = ", ".join(clean_text(a.get("name")) for a in work.get("authors", [])[:5])
        external_ids = work.get("externalIds") or {}
        doi = external_ids.get("DOI", "")
        papers.append(
            Paper(
                source="SemanticScholar",
                topic=topic,
                query=query,
                title=clean_text(work.get("title")),
                year=str(work.get("year") or ""),
                venue=clean_text(work.get("venue")),
                authors=authors,
                doi=clean_text(doi),
                url=clean_text(work.get("url")),
                abstract=clean_text(work.get("abstract")),
                citations=str(work.get("citationCount") or ""),
            )
        )
    return papers


def dedupe(papers: Iterable[Paper]) -> list[Paper]:
    seen = set()
    unique = []
    for paper in papers:
        key = (paper.doi.lower(), paper.title.lower())
        if key in seen or not paper.title:
            continue
        seen.add(key)
        unique.append(paper)
    return unique


def relevance_score(paper: Paper) -> int:
    text = " ".join([paper.title, paper.venue, paper.abstract]).lower()
    terms = TOPIC_TERMS.get(paper.topic, [])
    return sum(1 for term in terms if term.lower() in text)


def filter_relevant(papers: Iterable[Paper], min_score: int) -> list[Paper]:
    if min_score <= 0:
        return list(papers)
    return [paper for paper in papers if relevance_score(paper) >= min_score]


def write_csv(path: Path, papers: list[Paper]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Paper.__dataclass_fields__.keys()))
        writer.writeheader()
        for paper in papers:
            writer.writerow(paper.__dict__)


def write_markdown(path: Path, papers: list[Paper]) -> None:
    lines = [
        "# Literature Search Results",
        "",
        "Generated by `tools/literature_scraper.py`.",
        "",
    ]
    current_topic = None
    for paper in papers:
        if paper.topic != current_topic:
            current_topic = paper.topic
            lines.extend([f"## {current_topic}", ""])
        abstract = paper.abstract[:500] + ("..." if len(paper.abstract) > 500 else "")
        lines.extend(
            [
                f"### {paper.title}",
                f"- Source: {paper.source}",
                f"- Query: {paper.query}",
                f"- Year/Venue: {paper.year} / {paper.venue}",
                f"- Authors: {paper.authors}",
                f"- Citations: {paper.citations}",
                f"- DOI: {paper.doi}",
                f"- URL: {paper.url}",
                f"- Abstract: {abstract}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/literature")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument(
        "--min-score",
        type=int,
        default=2,
        help="Minimum simple keyword relevance score; use 0 to disable filtering.",
    )
    parser.add_argument(
        "--source",
        choices=["openalex", "semantic_scholar", "both"],
        default="both",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    papers: list[Paper] = []
    for topic, queries in QUERIES.items():
        for query in queries:
            if args.source in {"openalex", "both"}:
                try:
                    papers.extend(query_openalex(topic, query, args.limit))
                except (HTTPError, URLError, TimeoutError) as exc:
                    print(f"WARNING: OpenAlex failed for {query!r}: {exc}")
                time.sleep(args.sleep)
            if args.source in {"semantic_scholar", "both"}:
                try:
                    papers.extend(query_semantic_scholar(topic, query, args.limit))
                except (HTTPError, URLError, TimeoutError) as exc:
                    print(f"WARNING: Semantic Scholar failed for {query!r}: {exc}")
                time.sleep(args.sleep)

    papers = filter_relevant(dedupe(papers), args.min_score)
    papers.sort(key=lambda p: (p.topic, -(int(p.citations) if p.citations.isdigit() else 0)))

    write_csv(out_dir / "literature_results.csv", papers)
    write_markdown(out_dir / "literature_results.md", papers)
    print(f"Wrote {len(papers)} unique papers to {out_dir}")


if __name__ == "__main__":
    main()
