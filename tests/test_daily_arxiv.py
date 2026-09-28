import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone

from daily_arxiv import (
    run,
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

    def test_versioned_relative_figure_does_not_duplicate_id(self):
        html = '<figure><img src="2609.26236v1/figures/COVER_scenario.png"></figure>'
        self.assertEqual(
            extract_first_figure_image(html, "https://arxiv.org/html/2609.26236"),
            "https://arxiv.org/html/2609.26236v1/figures/COVER_scenario.png",
        )

    def test_repair_duplicate_id_preserves_version(self):
        paper = {"arxiv_id": "2609.26236", "introduction_image":
                 "https://arxiv.org/html/2609.26236/2609.26236v1/figures/COVER_scenario.png"}
        expected = "https://arxiv.org/html/2609.26236v1/figures/COVER_scenario.png"
        self.assertEqual(repair_cached_image_url(paper), expected)
        paper["introduction_image"] = expected
        self.assertEqual(repair_cached_image_url(paper), expected)

    def test_absolute_figure_url_is_preserved(self):
        for source in ["https://arxiv.org/html/2609.26236v1/x1.png",
                       "/html/2609.26236v1/x1.png"]:
            self.assertEqual(extract_first_figure_image(
                f'<figure><img src="{source}"></figure>',
                "https://arxiv.org/html/2609.26236"),
                "https://arxiv.org/html/2609.26236v1/x1.png")

    def test_partial_update_warning_is_visible(self):
        config = {"title": "Test", "topics": {"Example": {"terms": ["watermark"]}}}
        catalog = {"meta": {"failed_topics": ["Example"]}, "topics": {}}
        self.assertIn("latest update was incomplete", render_readme(config, catalog))

    def test_table_image_alt_escapes_pipe(self):
        preview = introduction_preview({"title": "A | B", "introduction_image": "https://example.com/x.png"})
        self.assertNotIn("|", preview)
        self.assertIn("&#124;", preview)

    def test_retry_failed_image_and_record_successful_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text('title: Test\ndata_path: data.json\nreadme_path: README.md\n'
                              'image_fetch_delay_seconds: 0\ntopics:\n  Test:\n    terms: [watermark]\n')
            paper = {"arxiv_id": "2609.26236", "title": "Test", "abs_url": "https://arxiv.org/abs/2609.26236",
                     "pdf_url": "https://arxiv.org/pdf/2609.26236", "first_seen": "2026-09-01",
                     "introduction_image": None}
            catalog = {"meta": {"last_updated": "2026-09-01T00:00:00+00:00"},
                       "topics": {"Test": {paper["arxiv_id"]: paper}}}
            data = root / "data.json"
            data.write_text(json.dumps(catalog))
            with patch("daily_arxiv.fetch_topic", return_value=[dict(paper)]), patch(
                "daily_arxiv.fetch_first_figure_image", return_value=None
            ) as fetch:
                self.assertEqual(run(config), 0)
                fetch.assert_called_once()
            result = json.loads(data.read_text())
            self.assertNotEqual(result["meta"]["last_updated"], catalog["meta"]["last_updated"])
            self.assertEqual(result["topics"]["Test"][paper["arxiv_id"]]["first_seen"], "2026-09-01")
            before = data.read_text()
            with self.assertLogs("watermarking-arxiv-daily", level="ERROR"), patch(
                "daily_arxiv.fetch_topic", side_effect=RuntimeError("unavailable")
            ):
                self.assertEqual(run(config), 1)
            self.assertEqual(data.read_text(), before)

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
