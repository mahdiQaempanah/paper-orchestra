"""Run PaperOrchestra model requests through the authenticated Codex CLI."""

import os
from pathlib import Path
import subprocess
import tempfile

import pymupdf


def call_codex_with_contents(
    contents, generation_configs, result_parsing_func, check_parsed_response_not_none
):
    """Return the same response structure as the existing Gemini adapter."""
    with tempfile.TemporaryDirectory(prefix="paper-orchestra-codex-") as directory:
        directory = Path(directory)
        prompt_parts = []
        images = []
        system_instruction = generation_configs.get("system_instruction")
        if system_instruction:
            prompt_parts.append(str(system_instruction))

        parts = contents if isinstance(contents, (list, tuple)) else [contents]
        for part in parts:
            if isinstance(part, str):
                prompt_parts.append(part)
                continue
            if getattr(part, "text", None) is not None:
                prompt_parts.append(part.text)
                continue
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None or inline_data.data is None:
                raise ValueError("Codex backend encountered an unsupported content part")
            mime_type = inline_data.mime_type or ""
            if mime_type.startswith("image/"):
                extension = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                }.get(mime_type)
                if extension is None:
                    raise ValueError(f"Unsupported image type: {mime_type}")
                image = directory / f"image-{len(images)}{extension}"
                image.write_bytes(inline_data.data)
                images.append(image)
            elif mime_type == "application/pdf":
                with pymupdf.open(stream=inline_data.data, filetype="pdf") as pdf:
                    for page in pdf:
                        image = directory / f"page-{len(images)}.png"
                        page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(
                            str(image)
                        )
                        images.append(image)
            else:
                raise ValueError(f"Unsupported content type: {mime_type}")

        output = directory / "response.txt"
        command = ["codex"]
        if generation_configs.get("tools"):
            command.append("--search")
        command.extend([
            "exec", "--ephemeral", "--skip-git-repo-check",
            "--sandbox", "read-only", "--output-last-message", str(output),
        ])
        if os.getenv("PAPER_ORCHESTRA_CODEX_MODEL"):
            command.extend(["--model", os.environ["PAPER_ORCHESTRA_CODEX_MODEL"]])
        for image in images:
            command.extend(["--image", str(image)])
        command.append("-")

        result = subprocess.run(
            command, input="\n\n".join(prompt_parts), text=True, capture_output=True
        )
        if result.returncode:
            raise RuntimeError(f"Codex CLI failed ({result.returncode}): {result.stderr.strip()}")
        if not output.exists():
            raise RuntimeError("Codex CLI did not produce a response")
        raw_response = output.read_text(encoding="utf-8")
        parsed_response = result_parsing_func(raw_response)
        if check_parsed_response_not_none and parsed_response is None:
            raise ValueError("Codex response could not be parsed")
        return {"raw_response": raw_response, "parsed_response": parsed_response}
