"""Build trends report: cluster recent papers, score by momentum, ask LLM
to identify research-industry gaps and startup theses for the top clusters.

Reads:
  - ../data/*_AI_enhanced_*.jsonl        (papers with AI summaries)
  - ../data/embeddings/*.jsonl           (sibling embedding files)

Writes:
  - ../data/trends/{date}.json
  - ../data/trends/trends-list.txt
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import dotenv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from trends_structure import TrendsReport

if os.path.exists(".env"):
    dotenv.load_dotenv()

SYSTEM_PROMPT = open("trends_system.txt", "r").read()

USER_TEMPLATE = """Corpus window: last {window_days} days, {paper_count} papers across {cluster_count} clusters.

Below are the top {top_n} clusters ranked by (size × growth). Analyze each and return your TrendsReport.

{clusters}
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--out-dir", default="../data/trends")
    p.add_argument("--window-days", type=int, default=90)
    p.add_argument("--n-clusters", type=int, default=20)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--report-date", default=None, help="YYYY-MM-DD; defaults to today UTC")
    return p.parse_args()


def date_from_filename(path: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    return m.group(1) if m else None


def load_papers(data_dir: str, language: str, window_days: int, end: datetime) -> Dict[str, dict]:
    cutoff = end - timedelta(days=window_days)
    pattern = os.path.join(data_dir, f"*_AI_enhanced_{language}.jsonl")
    by_id: Dict[str, dict] = {}
    for path in sorted(glob.glob(pattern)):
        date = date_from_filename(path)
        if not date:
            continue
        try:
            d = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if d < cutoff or d > end:
            continue
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = item.get("id")
                if not pid:
                    continue
                item["_date"] = date
                if pid not in by_id:
                    by_id[pid] = item
    return by_id


def load_embeddings(data_dir: str, paper_ids: set) -> Dict[str, np.ndarray]:
    by_id: Dict[str, np.ndarray] = {}
    pattern = os.path.join(data_dir, "embeddings", "*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec.get("id")
                vec = rec.get("v")
                if pid in paper_ids and vec:
                    by_id[pid] = np.asarray(vec, dtype=np.float32)
    return by_id


def cluster_papers(vectors: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    k = min(k, max(2, vectors.shape[0] // 5))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(vectors)


def cluster_keywords(corpus: List[str], labels: np.ndarray, top_k: int = 8) -> Dict[int, List[str]]:
    if not corpus:
        return {}
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    try:
        tfidf = vectorizer.fit_transform(corpus)
    except ValueError:
        return {}
    feature_names = np.array(vectorizer.get_feature_names_out())
    out: Dict[int, List[str]] = {}
    for cid in sorted(set(int(l) for l in labels)):
        mask = labels == cid
        if not mask.any():
            continue
        mean_tfidf = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:top_k]
        out[cid] = feature_names[top_idx].tolist()
    return out


def growth_ratio(dates: List[str], end: datetime) -> float:
    """Ratio of papers in last 4 weeks vs. previous 8 weeks. Smoothed."""
    recent_cut = end - timedelta(days=28)
    prior_cut = end - timedelta(days=84)
    recent = sum(1 for d in dates if datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc) >= recent_cut)
    prior = sum(
        1 for d in dates
        if prior_cut <= datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc) < recent_cut
    )
    return (recent + 1) / (max(prior, 1) / 2 + 1)


def build_cluster_summary(
    cid: int,
    paper_ids: List[str],
    papers: Dict[str, dict],
    keywords: List[str],
    growth: float,
    centroid: np.ndarray,
    embeddings: Dict[str, np.ndarray],
    sample_size: int = 8,
) -> dict:
    sims = []
    for pid in paper_ids:
        v = embeddings.get(pid)
        if v is None:
            continue
        denom = (np.linalg.norm(v) * np.linalg.norm(centroid)) or 1.0
        sims.append((float(np.dot(v, centroid) / denom), pid))
    sims.sort(reverse=True)
    sample_ids = [pid for _, pid in sims[:sample_size]]

    sample_blocks = []
    for pid in sample_ids:
        p = papers[pid]
        ai = p.get("AI") or {}
        sample_blocks.append(
            f"  - {pid} ({p.get('_date')}) {p.get('title', '').strip()}\n"
            f"    tldr: {ai.get('tldr', '')}"
        )

    return {
        "id": cid,
        "size": len(paper_ids),
        "growth_ratio": round(growth, 2),
        "score": round(len(paper_ids) * growth, 2),
        "keywords": keywords,
        "sample_paper_ids": sample_ids,
        "sample_blocks": sample_blocks,
        "all_paper_ids": paper_ids,
    }


def format_clusters_for_prompt(clusters: List[dict]) -> str:
    out = []
    for c in clusters:
        out.append(
            f"Cluster {c['id']}\n"
            f"  size: {c['size']} papers   growth_ratio: {c['growth_ratio']}   score: {c['score']}\n"
            f"  keywords: {', '.join(c['keywords'])}\n"
            f"  representative papers:\n" + "\n".join(c["sample_blocks"])
        )
    return "\n\n".join(out)


def main():
    args = parse_args()
    model_name = os.environ.get("TRENDS_MODEL_NAME") or os.environ.get("MODEL_NAME", "gpt-4o-mini")
    language = os.environ.get("LANGUAGE", "English")

    end = (
        datetime.strptime(args.report_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.report_date
        else datetime.now(timezone.utc)
    )
    end_str = end.strftime("%Y-%m-%d")

    papers = load_papers(args.data_dir, language, args.window_days, end)
    if len(papers) < 50:
        print(f"only {len(papers)} papers in window — too few to cluster meaningfully", file=sys.stderr)
        sys.exit(0)

    embeddings = load_embeddings(args.data_dir, set(papers.keys()))
    aligned_ids = [pid for pid in papers if pid in embeddings]
    if len(aligned_ids) < 50:
        print(f"only {len(aligned_ids)} papers have embeddings — run embed.py daily first", file=sys.stderr)
        sys.exit(0)

    print(f"clustering {len(aligned_ids)} papers ending {end_str}", file=sys.stderr)
    matrix = np.stack([embeddings[pid] for pid in aligned_ids])
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)

    labels = cluster_papers(matrix, args.n_clusters)

    corpus_texts = []
    for pid in aligned_ids:
        p = papers[pid]
        ai = p.get("AI") or {}
        corpus_texts.append(f"{p.get('title', '')} {ai.get('tldr', '')} {ai.get('method', '')}")
    keywords = cluster_keywords(corpus_texts, labels)

    cluster_to_papers: Dict[int, List[str]] = defaultdict(list)
    for pid, label in zip(aligned_ids, labels):
        cluster_to_papers[int(label)].append(pid)

    centroids: Dict[int, np.ndarray] = {}
    for cid, ids in cluster_to_papers.items():
        idxs = [aligned_ids.index(pid) for pid in ids]
        centroids[cid] = matrix[idxs].mean(axis=0)

    summaries: List[dict] = []
    for cid, ids in cluster_to_papers.items():
        if len(ids) < 5:
            continue
        dates = [papers[pid]["_date"] for pid in ids]
        growth = growth_ratio(dates, end)
        summaries.append(build_cluster_summary(
            cid, ids, papers, keywords.get(cid, []), growth, centroids[cid], embeddings,
        ))

    summaries.sort(key=lambda c: c["score"], reverse=True)
    top = summaries[: args.top_n]

    llm = ChatOpenAI(model=model_name).with_structured_output(TrendsReport, method="function_calling")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(USER_TEMPLATE),
    ])
    chain = prompt | llm

    print(f"asking {model_name} to analyze top {len(top)} clusters", file=sys.stderr)
    report: TrendsReport = chain.invoke({
        "language": language,
        "window_days": args.window_days,
        "paper_count": len(aligned_ids),
        "cluster_count": len(summaries),
        "top_n": len(top),
        "clusters": format_clusters_for_prompt(top),
    })

    cluster_lookup = {c["id"]: c for c in top}
    out_clusters = []
    for analysis in report.top_clusters:
        c = cluster_lookup.get(analysis.cluster_id)
        if not c:
            continue
        out_clusters.append({
            **analysis.model_dump(),
            "size": c["size"],
            "growth_ratio": c["growth_ratio"],
            "score": c["score"],
            "keywords": c["keywords"],
            "sample_paper_ids": c["sample_paper_ids"],
            "all_paper_ids": c["all_paper_ids"],
        })

    paper_index = {
        pid: {
            "title": papers[pid].get("title", "").strip(),
            "authors": papers[pid].get("authors", []),
            "abs": papers[pid].get("abs") or f"https://arxiv.org/abs/{pid}",
            "date": papers[pid].get("_date"),
            "categories": papers[pid].get("categories", []),
        }
        for pid in aligned_ids
    }

    output = {
        "report_date": end_str,
        "window_days": args.window_days,
        "paper_count": len(aligned_ids),
        "cluster_count": len(summaries),
        "language": language,
        "model": model_name,
        "overview": report.overview,
        "clusters": out_clusters,
        "paper_index": paper_index,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{end_str}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}", file=sys.stderr)

    list_path = os.path.join(args.out_dir, "trends-list.txt")
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.out_dir, "*.json")))
    with open(list_path, "w") as f:
        for name in files:
            f.write(name + "\n")
    print(f"wrote {list_path} with {len(files)} entries", file=sys.stderr)


if __name__ == "__main__":
    main()
