import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None


RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/golden_set.jsonl")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
PAIRS_PER_DOC = int(os.getenv("QA_PAIRS_PER_DOC", "12"))


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\\n", "", stripped)
        stripped = re.sub(r"\\n```$", "", stripped)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", stripped, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))

    if not isinstance(data, list):
        raise ValueError("Model output is not a JSON array")

    return [item for item in data if isinstance(item, dict)]


def _build_prompt(doc_name: str, text: str, num_pairs: int) -> str:
    return f"""
You are creating synthetic QA data for RAG evaluation.

Input document: {doc_name}
Generate exactly {num_pairs} objects in a JSON array.
Each object must follow this schema:
{{
  "question": "string",
  "expected_answer": "string",
  "context": "exact supporting quote or paraphrase from the document",
  "metadata": {{
    "difficulty": "easy|medium|hard",
    "type": "fact-check|reasoning|adversarial|out-of-context"
  }}
}}

Rules:
1) Your `question`, `expected_answer`, `context` must be in Vietnamese
2) Use only facts from the provided document for non-edge cases
3) But you can create edge cases which ask out of context, in those case, `context` must be empty string, type is out-of-context
4) Include at least one adversarial question that is tricky but still answerable from the document.
5) Keep expected_answer concise and directly verifiable.
6) Return JSON only. No markdown fences, no extra keys.

Document content:
{text}
""".strip()


def _generate_with_gemini(client: "genai.Client", prompt: str) -> str:
    base_config = {
        "temperature": 0.9,
        "response_mime_type": "application/json",
        "max_output_tokens": 4096,
    }

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={**base_config, "service_tier": "flex"},
        )
    except Exception:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=base_config,
        )

    if not response or not getattr(response, "text", None):
        raise ValueError("Gemini returned empty response")

    return response.text


async def generate_qa_from_text(client: "genai.Client", doc_name: str, doc_id: int,text: str, num_pairs: int = 5) -> List[Dict[str, Any]]:
    prompt = _build_prompt(doc_name=doc_name, text=text, num_pairs=num_pairs)
    model_text = await asyncio.to_thread(_generate_with_gemini, client, prompt)
    pairs = _extract_json_array(model_text)

    normalized: List[Dict[str, Any]] = []
    for pair in pairs:
        if not all(key in pair for key in ["question", "expected_answer", "context"]):
            continue
        pair.setdefault("metadata", {})
        pair["metadata"]["source_doc"] = doc_name
        pair["metadata"]["doc_id"] = doc_id
        normalized.append(pair)

    return normalized


async def main() -> None:
    if genai is None:
        raise ImportError("google-genai is not installed. Run: pip install google-genai")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running this script.")

    docs = sorted(RAW_DIR.glob("*.md"))
    if not docs:
        raise FileNotFoundError(f"No .md files found in {RAW_DIR}")

    client = genai.Client(api_key=api_key)

    all_pairs: List[Dict[str, Any]] = []
    for ith, doc_path in enumerate(docs):
        text = doc_path.read_text(encoding="utf-8")
        print(f"Generating from {doc_path.name}...")
        pairs = await generate_qa_from_text(
            client=client,
            doc_name=doc_path.name,
            doc_id = ith,
            text=text,
            num_pairs=PAIRS_PER_DOC,
        )
        all_pairs.extend(pairs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False, indent=4) + "\\n")

    print(f"Done! Saved {len(all_pairs)} QA pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
