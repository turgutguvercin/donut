"""Verified article boundaries, deterministic source allocation, and source audit."""
from collections import Counter, defaultdict
import hashlib
import json
import random

NATURAL = ("ascii_word", "turkish_word", "short_line", "medium_line", "long_sentence", "extra_long_sentence")


def load_articles(corpus):
    text = corpus.read_text(encoding="utf-8")
    manifest = json.loads(corpus.with_suffix(".sources.json").read_text(encoding="utf-8"))
    if hashlib.sha256(text.encode()).hexdigest() != manifest["sha256"]:
        raise ValueError("Corpus checksum mismatch")
    articles, titles, revisions = [], set(), set()
    previous = 0
    for source in manifest["pages"]:
        start, end = source["character_start"], source["character_end"]
        part = text[start:end]
        if start != previous or end <= start or text[end:end + 2] != " \n":
            raise ValueError("Invalid article boundaries")
        if hashlib.sha256(part.encode()).hexdigest() != source["sha256"]:
            raise ValueError("Article checksum mismatch")
        if source["title"] in titles or source["revision"] in revisions or source["split"] not in ("train", "val", "test"):
            raise ValueError("Duplicate article or invalid split")
        articles.append({**source, "text": part})
        titles.add(source["title"])
        revisions.add(source["revision"])
        previous = end + 2
    if previous != len(text):
        raise ValueError("Unattributed corpus text")
    return articles


def assign_articles(plan, articles, seed):
    """Round-robin shuffled articles within topics, independently per split/type.

    At 100k with 20 topics and 18/1/1 articles per topic: exactly 25 target
    samples per article per natural content type, or 150 per article overall.
    Synthetic rows also get same-split articles for neighboring context only.
    """
    available = defaultdict(lambda: defaultdict(list))
    for i, article in enumerate(articles):
        available[article["split"]][article["topic"]].append(i)
    rng = random.Random(seed + 400001)
    groups = defaultdict(list)
    for row in plan:
        groups[row["split"], row["content_type"]].append(row)
    for (split, kind), rows in sorted(groups.items()):
        topics = sorted(available[split])
        if not topics:
            raise ValueError("No source articles for " + split)
        rng.shuffle(topics)
        queues = {}
        for topic in topics:
            queues[topic] = sorted(available[split][topic])
            rng.shuffle(queues[topic])
        order = list(rows)
        rng.shuffle(order)
        used = Counter()
        for i, row in enumerate(order):
            topic = topics[i % len(topics)]
            queue = queues[topic]
            row["article_index"] = queue[used[topic] % len(queue)]
            used[topic] += 1


def audit_sources(rows, manifest):
    pages = {p["revision"]: p for p in manifest["pages"]}
    if len(pages) != len(manifest["pages"]):
        raise ValueError("Duplicate article revisions")
    by_topic, by_article, by_split = Counter(), Counter(), defaultdict(Counter)
    by_kind = defaultdict(Counter)
    available = defaultdict(lambda: defaultdict(list))
    for p in pages.values():
        available[p["split"]][p["topic"]].append(p["revision"])
    for row in rows:
        kind, split = row["categories"]["content_type"], row["split"]
        for key in ("source_article", "context_article"):
            p = row[key]
            if p is None:
                if key == "context_article" or kind in NATURAL:
                    raise ValueError("Missing article provenance")
                continue
            expected = pages.get(p["revision"])
            required = {"title", "revision", "url", "sha256", "topic", "split"}
            if not required <= p.keys() or expected is None or any(p[k] != expected[k] for k in required) or p["split"] != split:
                raise ValueError("Article provenance/split mismatch")
        if kind not in NATURAL:
            if row["source_article"] is not None:
                raise ValueError("Synthetic text attributed to Wikipedia")
            continue
        p = row["source_article"]
        by_topic[p["topic"]] += 1
        by_article[str(p["revision"])] += 1
        by_split[split][p["topic"]] += 1
        by_kind[split, kind][p["revision"]] += 1
    for (split, kind), counts in by_kind.items():
        topic_counts = []
        for topic, revisions in available[split].items():
            amounts = [counts[r] for r in revisions]
            if max(amounts) - min(amounts) > 1:
                raise ValueError("Unbalanced article allocation")
            topic_counts.append(sum(amounts))
        if max(topic_counts) - min(topic_counts) > 1:
            raise ValueError("Unbalanced topic allocation")
    return {"passed": True, "natural_images": sum(by_topic.values()),
            "articles_used": len(by_article), "topics_used": len(by_topic),
            "article_disjoint_splits_including_context": True,
            "balanced_per_content_type_and_split": True,
            "images_per_topic": dict(sorted(by_topic.items())),
            "images_per_article_revision": dict(sorted(by_article.items())),
            "images_per_topic_by_split": {s: dict(sorted(c.items())) for s, c in sorted(by_split.items())},
            "article_counts_by_split": dict(Counter(p["split"] for p in pages.values())),
            "note": "Natural text only; dates, prices, IDs and confusable exercises remain separate. Common words can recur; word counts are not forced equal."}
