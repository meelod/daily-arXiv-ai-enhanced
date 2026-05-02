"""Embed AI-enhanced papers using OpenAI text-embedding-3-small.

Reads an _AI_enhanced_*.jsonl file produced by enhance.py and writes a sibling
embeddings file at ../data/embeddings/{date}.jsonl with one record per paper:
    {"id": "<arxiv id>", "v": [<1536 floats>]}

Cheap by design: only embeds (title + tldr) per paper.
"""
import argparse
import json
import os
import re
import sys
from typing import List

import dotenv
from openai import OpenAI

if os.path.exists(".env"):
    dotenv.load_dotenv()

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
BATCH_SIZE = 100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to _AI_enhanced_*.jsonl")
    p.add_argument("--out-dir", default="../data/embeddings")
    return p.parse_args()


def embed_text(item: dict) -> str:
    title = (item.get("title") or "").strip()
    tldr = ((item.get("AI") or {}).get("tldr") or "").strip()
    return f"{title}\n\n{tldr}" if tldr else title


def date_from_filename(path: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    if not m:
        raise ValueError(f"could not extract date from {path}")
    return m.group(1)


def main():
    args = parse_args()
    # Defensive: empty OPENAI_BASE_URL (unset secret → "") breaks the SDK.
    if not os.environ.get("OPENAI_BASE_URL", "").strip():
        os.environ.pop("OPENAI_BASE_URL", None)
    if not os.path.exists(args.data):
        print(f"input file not found: {args.data}", file=sys.stderr)
        sys.exit(0)

    items: List[dict] = []
    with open(args.data, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not items:
        print("no items to embed", file=sys.stderr)
        sys.exit(0)

    date = date_from_filename(args.data)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{date}.jsonl")

    client = OpenAI()
    texts = [embed_text(it) for it in items]

    print(f"embedding {len(texts)} papers with {EMBED_MODEL}", file=sys.stderr)

    with open(out_path, "w") as out:
        for start in range(0, len(texts), BATCH_SIZE):
            batch_items = items[start:start + BATCH_SIZE]
            batch_texts = texts[start:start + BATCH_SIZE]
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch_texts)
            except Exception as e:
                print(f"embedding batch {start} failed: {e}", file=sys.stderr)
                continue
            for it, datum in zip(batch_items, resp.data):
                out.write(json.dumps({
                    "id": it.get("id"),
                    "v": [round(x, 6) for x in datum.embedding],
                }) + "\n")
            print(f"  embedded {start + len(batch_items)}/{len(texts)}", file=sys.stderr)

    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
