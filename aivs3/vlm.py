"""
Zero-shot VLM inference with a structured JSON prompt (manuscript Section 4.2).

All models receive the same system prompt, user prompt and image. Temperature
is fixed at 0 for models that accept it; reasoning models use their provider's
reasoning setting instead. Each model writes one JSONL file, one record per
image, so interrupted runs resume where they stopped.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
import traceback
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image, ImageOps
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PROMPT_DIR = Path(__file__).resolve().parent.parent / "configs" / "prompts"


# =============================================================================
# Prompt and image helpers
# =============================================================================
def load_prompts(prompt_dir: Path = PROMPT_DIR) -> tuple[str, str]:
    system = (Path(prompt_dir) / "system_prompt.txt").read_text(encoding="utf-8").strip()
    user = (Path(prompt_dir) / "user_prompt.txt").read_text(encoding="utf-8").strip()
    return system, user


def list_images(folder: Path) -> List[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def encode_image_jpeg(path: Path, apply_exif: bool = False) -> tuple[str, bytes]:
    """Return (base64 string, raw bytes) of the image re-encoded as JPEG.

    By default the stored pixel orientation is sent, as in the reported runs.
    ``apply_exif=True`` rotates the image to its EXIF display orientation first.
    """
    with Image.open(path) as img:
        if apply_exif:
            img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        raw = buf.getvalue()
    return base64.standard_b64encode(raw).decode("utf-8"), raw


def parse_json_response(raw: Optional[str]) -> Optional[dict]:
    """Parse a JSON object from a model reply, tolerating markdown fences."""
    if not raw:
        return None
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def make_record(filename: str, parsed: Optional[dict], raw: Optional[str],
                model: str, error: str = "") -> dict:
    return {
        "filename": filename,
        "model": model,
        "result": parsed,
        "raw_output_text": raw,
        "status": "ok" if parsed is not None else "error",
        "error": error if parsed is None else "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def load_processed(jsonl_path: Path) -> set:
    """File names already answered successfully in a JSONL output."""
    done = set()
    if Path(jsonl_path).exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") == "ok":
                    done.add(row.get("filename"))
    return done


def load_jsonl_results(jsonl_path: Path) -> Dict[str, dict]:
    """Latest successful result per file name."""
    out: Dict[str, dict] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok" and row.get("result"):
                out[row["filename"]] = row["result"]
    return out


def save_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
# Provider clients
# =============================================================================
class VLMClient:
    """One configured model. ``cfg`` is one entry of configs/models.yaml."""

    def __init__(self, tag: str, cfg: dict, system_prompt: str, user_prompt: str,
                 defaults: dict, apply_exif: bool = False):
        self.tag = tag
        self.provider = cfg["provider"]
        self.model_id = cfg["model_id"]
        self.cfg = cfg
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_tokens = int(cfg.get("max_output_tokens", defaults.get("max_output_tokens", 2048)))
        self.retries = int(defaults.get("max_retries", 3))
        self.retry_sleep = float(defaults.get("retry_sleep_seconds", 5))
        self.apply_exif = apply_exif
        self._client = None
        self._gemini_config = None

    # -- client construction ---------------------------------------------------
    def connect(self) -> None:
        import os
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif self.provider == "google":
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
            thinking = self.cfg.get("thinking", {})
            thinking_cfg = None
            if "budget" in thinking:
                thinking_cfg = types.ThinkingConfig(thinking_budget=int(thinking["budget"]))
            elif "level" in thinking:
                thinking_cfg = types.ThinkingConfig(thinking_level=thinking["level"])
            self._gemini_config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.cfg.get("temperature", 0),
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
                thinking_config=thinking_cfg,
                http_options=types.HttpOptions(timeout=120_000),
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # -- single request --------------------------------------------------------
    def _request(self, image_path: Path) -> Optional[str]:
        b64, raw_bytes = encode_image_jpeg(image_path, apply_exif=self.apply_exif)

        if self.provider == "openai":
            kwargs = {}
            if "reasoning_effort" in self.cfg:
                kwargs["reasoning_effort"] = self.cfg["reasoning_effort"]
                kwargs["max_completion_tokens"] = self.max_tokens
            else:
                kwargs["temperature"] = self.cfg.get("temperature", 0)
                kwargs["max_tokens"] = self.max_tokens
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": self.user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]},
                ],
                **kwargs,
            )
            return response.choices[0].message.content if response.choices else None

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=self.cfg.get("temperature", 0),
                system=self.system_prompt,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": self.user_prompt},
                ]}],
            )
            return response.content[0].text if response.content else None

        if self.provider == "google":
            from google.genai import types
            part = types.Part.from_bytes(data=raw_bytes, mime_type="image/jpeg")
            response = self._client.models.generate_content(
                model=self.model_id, contents=[self.user_prompt, part], config=self._gemini_config)
            return response.text or None

        raise ValueError(f"Unknown provider: {self.provider}")

    def analyze(self, image_path: Path) -> dict:
        wait = self.retry_sleep
        for attempt in range(1, self.retries + 1):
            try:
                raw = self._request(image_path)
                parsed = parse_json_response(raw)
                error = "" if parsed is not None else ("unparsable JSON" if raw else "empty response")
                return make_record(image_path.name, parsed, raw, self.model_id, error=error)
            except Exception as exc:  # network, quota and API errors are retried
                if attempt == self.retries:
                    return make_record(image_path.name, None, None, self.model_id,
                                       error=f"[attempt {attempt}] {exc}")
                rate_limited = "429" in str(exc) or "quota" in str(exc).lower()
                wait = wait * 2 if rate_limited else self.retry_sleep * attempt
                time.sleep(wait)
        return make_record(image_path.name, None, None, self.model_id, error="no attempt made")


# =============================================================================
# Runner
# =============================================================================
class ModelWorker(threading.Thread):
    """Runs one model over all images and appends results to its JSONL file."""

    def __init__(self, client: VLMClient, images: List[Path], output_jsonl: Path,
                 position: int, skip_processed: bool = True):
        super().__init__(daemon=True)
        self.client = client
        self.images = images
        self.output_jsonl = output_jsonl
        self.position = position
        self.skip_processed = skip_processed
        self.stats = {"ok": 0, "error": 0, "skipped": 0}
        self.errors: List[str] = []

    def run(self) -> None:
        done = load_processed(self.output_jsonl) if self.skip_processed else set()
        bar = tqdm(self.images, desc=f"[{self.client.tag:<15}]", position=self.position,
                   leave=True, dynamic_ncols=True)
        for img in bar:
            if img.name in done:
                self.stats["skipped"] += 1
                continue
            try:
                record = self.client.analyze(img)
            except Exception:
                record = make_record(img.name, None, None, self.client.model_id,
                                     error=traceback.format_exc())
            save_jsonl(self.output_jsonl, record)
            key = "ok" if record["status"] == "ok" else "error"
            self.stats[key] += 1
            if key == "error":
                self.errors.append(f"{img.name}: {record['error']}")
            bar.set_postfix(ok=self.stats["ok"], err=self.stats["error"], skip=self.stats["skipped"])


def run_benchmark(clients: List[VLMClient], images: List[Path], output_dir: Path,
                  skip_processed: bool = True,
                  on_done: Optional[Callable[[ModelWorker], None]] = None) -> List[ModelWorker]:
    """Run all models in parallel (one thread per model)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    for pos, client in enumerate(clients):
        client.connect()
        workers.append(ModelWorker(client, images, output_dir / f"{client.tag}.jsonl", pos, skip_processed))
    for w in workers:
        w.start()
    for w in workers:
        w.join()
        if on_done:
            on_done(w)
    return workers
