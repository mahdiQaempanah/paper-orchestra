import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types as python_types
import unittest
from unittest.mock import patch

import pymupdf
from google.genai import types

from utils.codex_utils import call_codex_with_contents
from utils.gemini_utils import call_gemini_with_contents, call_gemini_with_images
from utils.llm_backend_utils import call_llm_with_text_prompt
from utils.openalex_utils import openalex_title_search
from methods.agents.literature_review_agent import HybridLiteratureAgent, PaperData
from methods.agents.content_refinement_agent import ContentRefinementAgent
import paper_writing_cli


class CodexBackendTests(unittest.TestCase):
    def test_text_images_pdf_search_and_parser(self):
        pdf = pymupdf.open()
        pdf.new_page().insert_text((30, 30), "PDF evidence")
        pdf_bytes = pdf.tobytes()
        pdf.close()

        def run(command, input, text, capture_output):
            self.assertIn("--search", command)
            self.assertIn("--ignore-user-config", command)
            for feature in ("plugins", "multi_agent", "shell_tool", "unified_exec", "code_mode_host"):
                self.assertIn(feature, command)
            self.assertIn("system instructions", input)
            self.assertIn("paper text", input)
            images = [command[i + 1] for i, arg in enumerate(command) if arg == "--image"]
            self.assertEqual(len(images), 2)
            self.assertTrue(all(Path(image).is_file() for image in images))
            output = Path(command[command.index("--output-last-message") + 1])
            output.write_text('{"answer": 42}')
            return subprocess.CompletedProcess(command, 0)

        with patch("utils.codex_utils.subprocess.run", side_effect=run):
            response = call_codex_with_contents(
                ["paper text", types.Part.from_bytes(data=b"image", mime_type="image/png"),
                 types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")],
                {"tools": ["search"], "system_instruction": "system instructions"},
                json.loads, True,
            )
        self.assertEqual(response["parsed_response"], {"answer": 42})

    def test_routes_direct_and_unified_calls_to_codex(self):
        response = {"raw_response": "{}", "parsed_response": {}}
        with patch.dict(os.environ, {"PAPER_ORCHESTRA_BACKEND": "codex"}), \
             patch("utils.codex_utils.call_codex_with_contents", return_value=response) as backend:
            self.assertEqual(call_gemini_with_contents(["hello"], "gemini"), response)
            self.assertEqual(call_llm_with_text_prompt("hello", "gpt-example"), response)
            self.assertEqual(backend.call_count, 2)

    def test_failure_is_reported(self):
        result = subprocess.CompletedProcess(["codex"], 1, stderr="authentication failed")
        with patch("utils.codex_utils.subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "authentication failed"):
                call_codex_with_contents(["test"], {}, str, True)

    def test_jpeg_is_passed_with_its_correct_media_type(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "figure.jpg"
            image.write_bytes(b"jpeg bytes")
            with patch.dict(os.environ, {"PAPER_ORCHESTRA_BACKEND": "codex"}), \
                 patch("utils.codex_utils.call_codex_with_contents") as backend:
                call_gemini_with_images("prompt", [str(image)], "gemini")
                parts = backend.call_args.kwargs["contents"]
                self.assertEqual(parts[1].inline_data.mime_type, "image/jpeg")

    def test_cli_exits_nonzero_when_pipeline_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            template = root / "template"
            template.mkdir()
            writer = python_types.ModuleType("methods.paper_writer")
            writer.write_single_paper = lambda **kwargs: False
            args = ["paper_writing_cli.py", "--raw_materials_dir", str(raw),
                    "--latex_template_dir", str(template), "--output_dir", str(root / "out")]
            with patch.object(sys, "argv", args), \
                 patch.dict(sys.modules, {"methods.paper_writer": writer}):
                with self.assertRaises(SystemExit) as error:
                    paper_writing_cli.main()
            self.assertEqual(error.exception.code, 1)

    def test_all_search_failures_abort_codex_literature_stage(self):
        def fail_search(*args):
            raise RuntimeError("search unavailable")

        agent = HybridLiteratureAgent.__new__(HybridLiteratureAgent)
        agent.max_workers = 1
        agent._safe_print = lambda message: None
        agent._process_search_task_google = fail_search
        tasks = [{"section": "Introduction", "focus": "test"}]
        with patch.dict(os.environ, {"PAPER_ORCHESTRA_BACKEND": "codex"}):
            with self.assertRaisesRegex(RuntimeError, "All Codex literature searches failed"):
                agent._execute_tasks_in_parallel(tasks, {}, {}, set(), "2026-11")

    def test_failed_discovery_is_not_silently_discarded(self):
        agent = HybridLiteratureAgent.__new__(HybridLiteratureAgent)
        agent._safe_print = lambda message: None
        agent._discover_candidates = unittest.mock.Mock(side_effect=RuntimeError("offline"))
        task = {"section": "Introduction", "focus": "test"}
        with patch.dict(os.environ, {"PAPER_ORCHESTRA_BACKEND": "codex"}):
            with self.assertRaisesRegex(RuntimeError, "Codex literature discovery failed"):
                agent._process_search_task_google(task, {}, "2026-11")
        self.assertEqual(agent._discover_candidates.call_count, 3)

    def test_openalex_metadata_and_title_matching(self):
        response = unittest.mock.Mock()
        response.json.return_value = {"results": [{
            "display_name": "A Test Paper", "publication_year": 2020,
            "publication_date": "2020-02-03", "cited_by_count": 2,
            "abstract_inverted_index": {"abstract": [1], "Test": [0]},
            "authorships": [{"author": {"display_name": "A. Author"}}],
            "primary_location": {"source": {"display_name": "Test Journal", "type": "journal"}},
            "type": "article", "biblio": {"volume": "1", "first_page": "2", "last_page": "4"},
        }]}
        with patch("utils.openalex_utils.requests.get", return_value=response):
            result = openalex_title_search("A Test Paper", 2020)
            self.assertIsNone(openalex_title_search("Completely Different", 2020))
        self.assertEqual(result["abstract"], "Test abstract")
        self.assertEqual(result["journal"]["name"], "Test Journal")
        self.assertEqual(result["authors"], [{"name": "A. Author"}])

    def test_literature_agent_uses_openalex_after_semantic_scholar_miss(self):
        agent = HybridLiteratureAgent.__new__(HybridLiteratureAgent)
        agent.registry_lock = threading.Lock()
        agent._safe_print = lambda message: None
        metadata = {
            "title": "A Test Paper", "year": 2020, "publicationDate": "2020-02-03",
            "authors": [{"name": "A. Author"}], "abstract": "Test abstract",
            "venue": "Test Journal", "journal": None, "citationCount": 1,
        }
        registry = {}
        candidate = {"title": "A Test Paper", "year": 2020, "reason": "example"}
        task = {"section": "Introduction"}
        with patch.dict(os.environ, {"PAPER_ORCHESTRA_BACKEND": "codex"}), \
             patch("methods.agents.literature_review_agent.s2_title_search", return_value=None), \
             patch("utils.openalex_utils.openalex_title_search", return_value=metadata) as fallback:
            agent._enrich_and_register_paper(candidate, task, registry, "2026-11")
        fallback.assert_called_once_with("A Test Paper", 2020)
        self.assertEqual(registry["atestpaper"].title, "A Test Paper")
        self.assertEqual(registry["atestpaper"].metadata_source, "OpenAlex")

    def test_openalex_preprint_does_not_create_conference_citation(self):
        paper = PaperData(
            citation_key="test2020", title="Test", authors=["A. Author"],
            venue="arXiv", year=2020, abstract="Test", citation_count=0,
            found_in_section="Introduction", reason="Test", bibtex_type="misc",
            doi="10.1234/test",
        )
        agent = HybridLiteratureAgent.__new__(HybridLiteratureAgent)
        bibtex = agent._generate_bibtex([paper])
        self.assertIn("@misc{test2020", bibtex)
        self.assertIn("doi={10.1234/test}", bibtex)
        self.assertNotIn("booktitle", bibtex)

    def test_failed_peer_review_is_reported(self):
        agent = ContentRefinementAgent.__new__(ContentRefinementAgent)
        with patch.dict(os.environ, {"PAPER_ORCHESTRA_BACKEND": "codex"}), \
             patch("methods.agents.content_refinement_agent.load_paper", side_effect=RuntimeError("review unavailable")):
            with self.assertRaisesRegex(RuntimeError, "review unavailable"):
                agent._get_peer_review("test.pdf", "initial")


if __name__ == "__main__":
    unittest.main()
