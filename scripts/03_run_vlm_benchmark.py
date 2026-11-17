"""
Step 3 - Zero-shot VLM inference on every image (manuscript Section 4.2).

Runs the selected models in parallel (one thread per model) and writes one
JSONL file per model to ``--out``. Re-running skips images that already have a
successful answer.

API keys are read from a ``.env`` file in the repository root (see .env.example).

Examples
--------
python scripts/03_run_vlm_benchmark.py --input crop                       # all models, leaf crops
python scripts/03_run_vlm_benchmark.py --input full --models GPT_5.2 Flash_Lite_3.1
python scripts/03_run_vlm_benchmark.py --input crop --limit 5             # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aivs3.vlm import VLMClient, list_images, load_prompts, run_benchmark  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", choices=["crop", "full"], default="crop",
                    help="send detected leaf crops or full images")
    ap.add_argument("--crops", default="data/crops", help="folder written by step 2b")
    ap.add_argument("--images", default="data/images", help="folder of full images")
    ap.add_argument("--models", nargs="*", help="model tags from configs/models.yaml (default: all)")
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--prompts", default="configs/prompts")
    ap.add_argument("--out", default="outputs/vlm_raw")
    ap.add_argument("--limit", type=int, help="only the first N images")
    ap.add_argument("--apply-exif", action="store_true",
                    help="rotate full images to their EXIF display orientation before sending")
    ap.add_argument("--no-resume", action="store_true", help="re-run images that already have an answer")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tags = args.models or list(cfg["models"])
    unknown = [t for t in tags if t not in cfg["models"]]
    if unknown:
        sys.exit(f"Unknown model tag(s): {unknown}. Available: {list(cfg['models'])}")

    folder = Path(args.crops if args.input == "crop" else args.images)
    images = list_images(folder)
    if args.limit:
        images = images[: args.limit]
    if not images:
        sys.exit(f"No images found in {folder}")

    system_prompt, user_prompt = load_prompts(Path(args.prompts))
    clients = [VLMClient(t, cfg["models"][t], system_prompt, user_prompt, cfg.get("defaults", {}),
                         apply_exif=args.apply_exif) for t in tags]
    out_dir = Path(args.out) / args.input

    print(f"Images: {len(images)} ({args.input}) from {folder}")
    print(f"Models: {', '.join(tags)}")
    print(f"Output: {out_dir}\n")
    workers = run_benchmark(clients, images, out_dir, skip_processed=not args.no_resume)

    print("\nSummary")
    for w in workers:
        s = w.stats
        print(f"  {w.client.tag:<16} ok={s['ok']:>5}  error={s['error']:>4}  skipped={s['skipped']:>5}")
        if w.errors:
            log = out_dir / f"{w.client.tag}_errors.log"
            log.write_text("\n".join(w.errors), encoding="utf-8")
            print(f"  {'':<16} errors written to {log}")


if __name__ == "__main__":
    main()
