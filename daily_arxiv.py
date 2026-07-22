#!/usr/bin/env python3
"""Fetch watermarking papers from arXiv and regenerate README.md."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from html import escape as escape_html
from html.parser import HTMLParser
from pathlib import Path
from time import sleep
from typing import Any, Iterable
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import yaml


LOGGER = logging.getLogger("watermarking-arxiv-daily")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    required = {"title", "topics", "data_path", "readme_path"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    if not isinstance(config["topics"], dict) or not config["topics"]:
        raise ValueError("config.yaml must define at least one topic")
    for name, topic in config["topics"].items():
        terms = topic.get("terms", []) if isinstance(topic, dict) else []
        if not terms:
            raise ValueError(f"Topic {name!r} must contain at least one search term")
    return config


def normalize_arxiv_id(value: str) -> str:
    """Remove a trailing arXiv version suffix while preserving old-style IDs."""
    return re.sub(r"v\d+$", "", value.strip())


def build_query(terms: Iterable[str]) -> str:
    """Build an arXiv query that searches exact phrases in all metadata fields."""
    clauses: list[str] = []
    for term in terms:
        cleaned = " ".join(str(term).split()).replace('"', r'\"')
        if cleaned:
            clauses.append(f'all:"{cleaned}"')
    if not clauses:
        raise ValueError("At least one non-empty search term is required")
    return " OR ".join(clauses)


class FirstFigureImageParser(HTMLParser):
    """Find the first paper figure image in arXiv's HTML rendering."""

    def __init__(self) -> None:
        super().__init__()
        self.figure_depth = 0
        self.image_src: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "figure":
            self.figure_depth += 1
            return
        if tag != "img" or self.image_src:
            return

        attributes = dict(attrs)
        source = attributes.get("src") or attributes.get("data-src")
        classes = set((attributes.get("class") or "").split())
        if not source or source.startswith("data:"):
            return
        if self.figure_depth or "ltx_graphics" in classes:
            self.image_src = source

    def handle_endtag(self, tag: str) -> None:
        if tag == "figure" and self.figure_depth:
            self.figure_depth -= 1


def extract_first_figure_image(html: str, base_url: str) -> str | None:
    parser = FirstFigureImageParser()
    parser.feed(html)
    if not parser.image_src:
        return None
    return urljoin(base_url, parser.image_src)


def fetch_first_figure_image(arxiv_id: str, timeout: float = 15.0) -> str | None:
    """Fetch the first figure image URL from an arXiv HTML paper."""
    html_url = f"https://arxiv.org/html/{arxiv_id}"
    request = Request(
        html_url,
        headers={
            "User-Agent": "Watermarking-arxiv-daily/1.0 "
            "(+https://github.com/ai-kunkun/Watermarking-arxiv-daily)"
        },
    )
    with urlopen(request, timeout=timeout) as response:
        document = response.read(5_000_000).decode("utf-8", errors="replace")
        return extract_first_figure_image(document, response.geturl())


def enrich_papers_with_images(
    papers: list[dict[str, Any]],
    image_cache: dict[str, str | None],
    timeout: float,
    delay_seconds: float,
) -> None:
    """Add a cached first-figure URL to every paper record."""
    fetched = 0
    for paper in papers:
        paper_id = paper["arxiv_id"]
        if paper_id not in image_cache:
            try:
                image_cache[paper_id] = fetch_first_figure_image(paper_id, timeout)
            except Exception as error:
                LOGGER.warning("Could not fetch first image for %s: %s", paper_id, error)
                image_cache[paper_id] = None
            fetched += 1
            if fetched % 25 == 0:
                LOGGER.info("Fetched first-image previews: %d", fetched)
            if delay_seconds > 0:
                sleep(delay_seconds)
        paper["introduction_image"] = image_cache[paper_id]


def empty_catalog() -> dict[str, Any]:
    return {
        "meta": {"schema_version": 1, "last_updated": None},
        "topics": {},
    }


def load_catalog(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return empty_catalog()
    with path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    if not isinstance(catalog, dict):
        raise ValueError(f"Invalid catalog in {path}")
    catalog.setdefault("meta", {"schema_version": 1, "last_updated": None})
    catalog.setdefault("topics", {})
    return catalog


def paper_from_result(result: Any) -> dict[str, Any]:
    paper_id = normalize_arxiv_id(result.get_short_id())
    return {
        "arxiv_id": paper_id,
        "title": " ".join(result.title.split()),
        "authors": [str(author) for author in result.authors],
        "published": result.published.date().isoformat(),
        "updated": result.updated.date().isoformat(),
        "primary_category": result.primary_category,
        "abstract": " ".join(result.summary.split()),
        "journal_reference": getattr(result, "journal_ref", None),
        "comment": getattr(result, "comment", None),
        "abs_url": f"https://arxiv.org/abs/{paper_id}",
        "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
    }


def fetch_topic(terms: Iterable[str], max_results: int) -> list[dict[str, Any]]:
    # Imported lazily so formatting and unit tests can run independently.
    import arxiv

    query = build_query(terms)
    LOGGER.info("arXiv query: %s", query)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=3.0, num_retries=3)
    return [paper_from_result(result) for result in client.results(search)]


def merge_topic(existing: dict[str, Any], papers: Iterable[dict[str, Any]]) -> int:
    """Merge papers by arXiv ID and return the number of changed records."""
    changes = 0
    for paper in papers:
        paper_id = paper["arxiv_id"]
        if existing.get(paper_id) != paper:
            existing[paper_id] = paper
            changes += 1
    return changes


def escape_markdown(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ").strip()


def format_authors(authors: list[str], limit: int = 3) -> str:
    shown = authors[:limit]
    rendered = ", ".join(escape_markdown(author) for author in shown)
    if len(authors) > limit:
        rendered += " et al."
    return rendered


def topic_anchor(name: str) -> str:
    return re.sub(r"[^a-z0-9 -]", "", name.lower()).replace(" ", "-")


def compact_introduction(value: Any, limit: int = 300) -> str:
    """Return a compact, table-friendly introduction derived from the abstract."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return escape_markdown(text)

    candidate = text[: limit + 1]
    boundaries = [candidate.rfind(mark) for mark in (". ", "? ", "! ")]
    boundary = max(boundaries)
    if boundary >= int(limit * 0.25):
        shortened = candidate[: boundary + 1]
    else:
        shortened = text[:limit].rstrip(" ,;:-") + "…"
    return escape_markdown(shortened)


def introduction_preview(paper: dict[str, Any]) -> str:
    image_url = paper.get("introduction_image")
    if not image_url:
        return "—"
    title = escape_html(str(paper.get("title") or "Paper figure"), quote=True)
    source = escape_html(str(image_url), quote=True)
    return f'<img width="400" alt="{title}" src="{source}">'


def venue_and_year(paper: dict[str, Any]) -> str:
    """Prefer publication metadata and fall back to arXiv plus category."""
    venue = paper.get("journal_reference")
    if not venue:
        comment = " ".join(str(paper.get("comment") or "").split())
        match = re.search(
            r"(?:accepted (?:to|at)|to appear (?:at|in)|published (?:at|in))\s+([^.;]+)",
            comment,
            flags=re.IGNORECASE,
        )
        if match:
            venue = match.group(1).strip()
    if venue:
        venue = re.split(
            r"\s*[,;]?\s*(?:code|project page|website|implementation)\s+"
            r"(?:is\s+)?(?:at|available(?:\s+at)?)\b",
            " ".join(str(venue).split()),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        venue = re.sub(r"\s*https?://\S+.*$", "", venue).strip(" ,;:-")
    if not venue:
        category = paper.get("primary_category") or "preprint"
        venue = f"arXiv · {category}"
    venue_year_match = re.search(r"\b(?:19|20)\d{2}\b", str(venue))
    year = (
        venue_year_match.group(0)
        if venue_year_match
        else str(paper.get("published") or paper.get("updated") or "")[:4] or "—"
    )
    return f"{escape_markdown(venue)}<br>**{escape_markdown(year)}**"


def render_readme(config: dict[str, Any], catalog: dict[str, Any]) -> str:
    topic_data = catalog.get("topics", {})
    unique_paper_ids = {
        paper_id
        for papers in topic_data.values()
        for paper_id in papers
    }
    lines = [
        f"# 🛡️ {config['title']}",
        "",
        str(config.get("description", "")),
        "",
        "> Updated automatically from arXiv. **Introduction** shows the first figure "
        "found in each paper's arXiv HTML page.",
        "",
    ]

    last_updated = catalog.get("meta", {}).get("last_updated")
    if last_updated:
        timestamp = datetime.fromisoformat(last_updated).astimezone(
            ZoneInfo(config.get("timezone", "UTC"))
        )
        lines.extend(
            [
                f"**Last updated:** {timestamp:%Y-%m-%d %H:%M %Z} &nbsp; · &nbsp; "
                f"**Indexed:** {len(unique_paper_ids)} unique papers",
                "",
            ]
        )

    lines.extend(["<details>", '<summary><b>Search scope and keywords</b></summary>', ""])
    for topic_name, topic_config in config["topics"].items():
        terms = ", ".join(f"`{escape_markdown(term)}`" for term in topic_config["terms"])
        lines.append(f"- **{topic_name}:** {terms}")
    lines.extend(["", "</details>", ""])

    for topic_name, topic_config in config["topics"].items():
        papers = topic_data.get(topic_name, {})
        lines.extend(
            [
                f"## {topic_name} · {len(papers)} papers",
                "",
                "| **Title & Authors** | **Venue/Year** | **Introduction** | **Links** |",
                "|:---|:---:|:---|:---:|",
            ]
        )
        ordered = sorted(
            papers.values(),
            key=lambda paper: (paper.get("updated", ""), paper.get("published", ""), paper.get("arxiv_id", "")),
            reverse=True,
        )
        if not ordered:
            lines.append("| No matching papers yet | — | — | — |")
        for paper in ordered:
            title = escape_markdown(paper["title"])
            title_and_authors = (
                f"**{title}**<br><sub>{format_authors(paper.get('authors', []))}</sub>"
            )
            introduction = introduction_preview(paper)
            links = (
                f"[Paper]({paper['abs_url']})<br>"
                f"[PDF]({paper['pdf_url']})<br>"
                f"`{escape_markdown(paper['arxiv_id'])}`"
            )
            lines.append(
                f"| {title_and_authors} | {venue_and_year(paper)} | {introduction} | {links} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Automation",
            "",
            "GitHub Actions runs once per day and can also be started manually from the Actions tab. "
            "The workflow uses the free arXiv API and does not call a paid AI service.",
            "",
            "Inspired by [liutaocode/Video-Generation-arxiv-daily]"
            "(https://github.com/liutaocode/Video-Generation-arxiv-daily).",
            "",
        ]
    )
    return "\n".join(lines)


def write_text_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def save_catalog(path: Path, catalog: dict[str, Any]) -> bool:
    content = json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return write_text_if_changed(path, content)


def run(config_path: Path, render_only: bool = False) -> int:
    config = load_config(config_path)
    root = config_path.resolve().parent
    data_path = root / config["data_path"]
    readme_path = root / config["readme_path"]
    catalog = load_catalog(data_path)
    catalog_topics = catalog.setdefault("topics", {})
    image_cache: dict[str, str | None] = {
        paper_id: paper.get("introduction_image")
        for papers in catalog_topics.values()
        for paper_id, paper in papers.items()
        if "introduction_image" in paper
    }

    total_changes = 0
    successful_topics = 0
    if not render_only:
        normal_max = int(config.get("max_results_per_topic", 50))
        bootstrap_max = int(config.get("bootstrap_max_results_per_topic", normal_max))
        for topic_name, topic_config in config["topics"].items():
            existing = catalog_topics.setdefault(topic_name, {})
            max_results = bootstrap_max if not existing else normal_max
            try:
                papers = fetch_topic(topic_config["terms"], max_results)
            except Exception:
                LOGGER.exception("Failed to update topic: %s", topic_name)
                continue
            enrich_papers_with_images(
                papers,
                image_cache,
                timeout=float(config.get("image_fetch_timeout_seconds", 15)),
                delay_seconds=float(config.get("image_fetch_delay_seconds", 0.75)),
            )
            successful_topics += 1
            changes = merge_topic(existing, papers)
            total_changes += changes
            LOGGER.info("%s: fetched=%d changed=%d total=%d", topic_name, len(papers), changes, len(existing))

        if successful_topics == 0:
            LOGGER.error("All arXiv queries failed; existing files were left unchanged")
            return 1
        if total_changes:
            catalog.setdefault("meta", {})["schema_version"] = 1
            catalog["meta"]["last_updated"] = datetime.now(timezone.utc).isoformat()

    catalog_changed = save_catalog(data_path, catalog)
    readme_changed = write_text_if_changed(readme_path, render_readme(config, catalog))
    LOGGER.info(
        "Done: paper_changes=%d catalog_written=%s readme_written=%s",
        total_changes,
        catalog_changed,
        readme_changed,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Regenerate README.md from existing JSON without calling arXiv",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run(args.config, render_only=args.render_only)
    except Exception:
        LOGGER.exception("Update failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
