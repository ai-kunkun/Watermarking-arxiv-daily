import unittest
from datetime import datetime, timezone

from daily_arxiv import (
    build_query,
    compact_introduction,
    escape_markdown,
    extract_first_figure_image,
    format_authors,
    introduction_preview,
    merge_topic,
    migrate_catalog,
    normalize_arxiv_id,
    repair_cached_image_url,
    render_readme,
    venue_and_year,
)


class DailyArxivTests(unittest.TestCase):
    def test_normalize_arxiv_id(self):
        self.assertEqual(normalize_arxiv_id("2401.12345v3"), "2401.12345")
        self.assertEqual(normalize_arxiv_id("cs/9901001v2"), "cs/9901001")

    def test_build_query(self):
        self.assertEqual(
            build_query(["image watermarking", "LLM watermarking"]),
            'all:"image watermarking" OR all:"LLM watermarking"',
        )

    def test_merge_topic_is_idempotent(self):
        paper = {"arxiv_id": "2401.12345", "title": "Example"}
        existing = {}
        self.assertEqual(merge_topic(existing, [paper]), 1)
        self.assertEqual(merge_topic(existing, [paper]), 0)

    def test_markdown_escaping_and_authors(self):
        self.assertEqual(escape_markdown("A | B"), r"A \| B")
        self.assertEqual(format_authors(["A", "B", "C", "D"]), "A, B, C et al.")

    def test_compact_introduction(self):
        text = "A short first sentence. " + ("More detail " * 40)
        self.assertEqual(compact_introduction(text, limit=80), "A short first sentence.")

    def test_extract_first_figure_image(self):
        html = '<img src="logo.png"><figure><img class="ltx_graphics" src="x1.png"></figure>'
        self.assertEqual(
            extract_first_figure_image(html, "https://arxiv.org/html/2501.00001"),
            "https://arxiv.org/html/2501.00001/x1.png",
        )

    def test_repair_cached_image_url(self):
        paper = {
            "arxiv_id": "2602.15364",
            "introduction_image": "https://arxiv.org/html/x1.png",
        }
        self.assertEqual(
            repair_cached_image_url(paper),
            "https://arxiv.org/html/2602.15364/x1.png",
        )

        catalog = {
            "meta": {"last_updated": "2026-07-22T06:00:00+00:00"},
            "topics": {"Example": {"2602.15364": paper}},
        }
        self.assertEqual(migrate_catalog(catalog, "Asia/Shanghai"), 2)
        self.assertEqual(paper["first_seen"], "2026-07-22")

    def test_introduction_preview_uses_image(self):
        paper = {"title": "A & B", "introduction_image": "https://arxiv.org/html/x1.png"}
        preview = introduction_preview(paper)
        self.assertIn('width="400"', preview)
        self.assertIn('alt="A &amp; B"', preview)
        self.assertEqual(introduction_preview({}), "—")

    def test_venue_and_year(self):
        paper = {
            "journal_reference": "Proceedings of ExampleConf",
            "published": "2025-06-01",
            "primary_category": "cs.CR",
        }
        self.assertEqual(venue_and_year(paper), "Proceedings of ExampleConf<br>**2025**")

        comment_only = {
            "journal_reference": None,
            "comment": "Accepted to CVPR 2026, Code is at https://github.com/example/repo",
            "published": "2025-12-01",
            "primary_category": "cs.CV",
        }
        self.assertEqual(venue_and_year(comment_only), "CVPR 2026<br>**2026**")

    def test_readme_uses_requested_columns(self):
        config = {
            "title": "Watermarking arXiv Daily",
            "description": "Example",
            "repository": "ai-kunkun/Watermarking-arxiv-daily",
            "timezone": "UTC",
            "topics": {"LLM Watermarking": {"terms": ["LLM watermarking"]}},
        }
        catalog = {"meta": {"last_updated": None}, "topics": {"LLM Watermarking": {}}}
        readme = render_readme(config, catalog)
        self.assertIn(
            "| **Title & Authors** | **Venue/Year** | **Introduction** | **Links** |",
            readme,
        )
        self.assertIn("[LLM Watermarking](#llm-watermarking)", readme)

    def test_readme_lists_today_and_sorts_by_published_date(self):
        today = datetime.now(timezone.utc).date().isoformat()

        def paper(paper_id, title, published):
            return {
                "arxiv_id": paper_id,
                "title": title,
                "authors": ["Example Author"],
                "published": published,
                "updated": published,
                "primary_category": "cs.CR",
                "journal_reference": None,
                "comment": None,
                "abs_url": f"https://arxiv.org/abs/{paper_id}",
                "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
                "introduction_image": None,
                "first_seen": today,
            }

        config = {
            "title": "Watermarking arXiv Daily",
            "description": "Example",
            "timezone": "UTC",
            "topics": {"LLM Watermarking": {"terms": ["LLM watermarking"]}},
        }
        catalog = {
            "meta": {"last_updated": None},
            "topics": {
                "LLM Watermarking": {
                    "2501.00001": paper("2501.00001", "Older Paper", "2025-01-01"),
                    "2601.00001": paper("2601.00001", "Newer Paper", "2026-01-01"),
                }
            },
        }
        readme = render_readme(config, catalog)
        self.assertIn(f"## Today's additions · {today}", readme)
        self.assertIn("· 2 new papers", readme)
        table_start = readme.index("| **Title & Authors**")
        self.assertLess(
            readme.index("Newer Paper", table_start),
            readme.index("Older Paper", table_start),
        )


if __name__ == "__main__":
    unittest.main()
