from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from isc_knowledge_base.markdown_import import import_markdown


class MarkdownImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def test_imports_sections_with_source_paths_and_line_ranges(self) -> None:
        policies = self.root / "policies"
        policies.mkdir()
        (policies / "travel.MD").write_text(
            "Introduction.\n\n"
            "Travel policy\n=============\n\n"
            "## Booking\n\n"
            "Book before Friday.\n\n"
            "```text\n# This stays in the body\n```\n\n"
            "> ## This is a quote\n\n"
            "## Reimbursement\n\n"
            "Keep the receipts.\n",
            encoding="utf-8",
        )
        (self.root / "notes.txt").write_text("Other format.", encoding="utf-8")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "ignored.md").write_text("Git metadata.", encoding="utf-8")

        commit_sha = "a" * 40
        fragments = list(import_markdown(self.root, project_id="12", commit_sha=commit_sha))
        self.assertEqual(
            [fragment.source.heading_path for fragment in fragments],
            [(), ("Travel policy", "Booking"), ("Travel policy", "Reimbursement")],
        )
        self.assertEqual(
            [(fragment.source.start_line, fragment.source.end_line) for fragment in fragments],
            [(1, 1), (6, 14), (16, 18)],
        )
        self.assertIn("# This stays in the body", fragments[1].text)
        self.assertIn("> ## This is a quote", fragments[1].text)
        self.assertEqual(fragments[1].source.path, "policies/travel.MD")
        self.assertEqual(fragments[1].source.project_id, "12")
        self.assertEqual(fragments[1].source.commit_sha, commit_sha)

    def test_ids_distinguish_repeated_headings_and_survive_body_edits(self) -> None:
        document = self.root / "travel.md"
        markdown = "# Policy\n\n## Rules\n\nFirst version.\n\n## Rules\n\nKeep receipts.\n"
        document.write_text(markdown, encoding="utf-8")
        first = list(import_markdown(self.root, project_id="12", commit_sha="a" * 40))
        repeat = list(import_markdown(self.root, project_id="12", commit_sha="a" * 40))
        self.assertEqual(first, repeat)
        self.assertEqual(first[0].source.heading_path, first[1].source.heading_path)
        self.assertNotEqual(first[0].id, first[1].id)

        document.write_text(markdown.replace("First version.", "New version."), encoding="utf-8")
        second = list(import_markdown(self.root, project_id="12", commit_sha="b" * 40))
        self.assertEqual([fragment.id for fragment in first], [fragment.id for fragment in second])
        self.assertEqual(second[0].text, "New version.")
        other_project = list(import_markdown(self.root, project_id="21"))
        self.assertNotEqual(second[0].id, other_project[0].id)
        self.assertIsNone(other_project[0].source.commit_sha)
