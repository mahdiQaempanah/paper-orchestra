import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pymupdf
from google.genai import types

from utils.codex_utils import call_codex_with_contents
from utils.gemini_utils import call_gemini_with_contents, call_gemini_with_images
from utils.llm_backend_utils import call_llm_with_text_prompt


class CodexBackendTests(unittest.TestCase):
    def test_text_images_pdf_search_and_parser(self):
        pdf = pymupdf.open()
        pdf.new_page().insert_text((30, 30), "PDF evidence")
        pdf_bytes = pdf.tobytes()
        pdf.close()

        def run(command, input, text, capture_output):
            self.assertIn("--search", command)
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


if __name__ == "__main__":
    unittest.main()
