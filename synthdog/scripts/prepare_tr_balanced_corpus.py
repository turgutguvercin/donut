#!/usr/bin/env python3
"""Prepare a diverse, revision-pinned Turkish Wikipedia corpus with article splits.

Cache text outside Git. The manifest carries attribution, byte hashes, topic,
split, and character boundaries; --manifest reproduces exactly those revisions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import threading
from urllib.error import HTTPError, URLError

from prepare_tr_corpus import API, fetch

REQUEST_LOCK = threading.Lock()
LAST_REQUEST = 0.0


def cached_fetch(page, cache):
    identity = str(page.get("revision", page["title"]))
    path = cache / (hashlib.sha256(identity.encode()).hexdigest() + ".json")
    paths = [path]
    if "revision" in page:
        paths.append(cache / (hashlib.sha256(page["title"].encode()).hexdigest() + ".json"))
    for cached in paths:
        if not cached.exists():
            continue
        item = json.loads(cached.read_text(encoding="utf-8"))
        if "revision" in page and item["source"]["revision"] != page["revision"]:
            if cached == path:
                raise ValueError("Cached revision mismatch")
            continue
        digest = hashlib.sha256(item["text"].encode()).hexdigest()
        if digest != item["source"]["sha256"] or page.get("sha256", digest) != digest:
            raise ValueError("Cached article checksum mismatch")
        return item["text"], item["source"]
    for attempt in range(4):
        try:
            # Shared pacing includes retries. Cached revisions need no request.
            global LAST_REQUEST
            with REQUEST_LOCK:
                time.sleep(max(0, 3.0 - (time.monotonic() - LAST_REQUEST)))
                LAST_REQUEST = time.monotonic()
                text, source = fetch(page)
            break
        except (HTTPError, URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 3:
                raise
            delay = 30 if isinstance(error, HTTPError) and error.code == 429 else 2 ** attempt
            if isinstance(error, HTTPError) and error.headers.get("Retry-After", "").isdigit():
                delay = max(delay, int(error.headers["Retry-After"]))
            print(f"Retrying {page['title']}: {error}; waiting {delay}s", flush=True)
            while delay:
                pause = min(30, delay)
                time.sleep(pause)
                delay -= pause
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps({"text": text, "source": source}, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return text, source


def eligibility(text, codepoints):
    # Validate all six classes, both case variants and the intersection of the
    # actual fonts' cmaps before committing to an article/split assignment.
    import generate_tr_dataset as d
    result = {}
    pools = d.make_pools(text, allow_empty=True)
    for split in ("train", "val", "test"):
        counts = {}
        for (kind, pool_split), values in pools.items():
            if pool_split != split:
                continue
            eligible = [v for v in values if set(map(ord, v + d.base.turkish_upper(v))) <= codepoints
                        and (kind != "ascii_word" or d.base.turkish_upper(v).isascii())]
            counts[kind] = len(eligible)
        if all(counts.values()):
            result[split] = counts
    return result


def write_corpus(output, records, extra=None, expected=None):
    texts, pages, offset = [], [], 0
    for text, source in records:
        pages.append({**source, "character_start": offset, "character_end": offset + len(text)})
        texts.append(text)
        offset += len(text) + 2
    corpus = " \n".join(texts) + " \n"
    manifest = {"schema_version": 2, "language": "tr", "source": API, "pages": pages,
                "license": "CC-BY-SA-4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "attribution": "Turkish Wikipedia contributors; see revision URLs and page histories.",
                "transformation": "HTML paragraphs, references removed, NFC normalization, whitespace collapsed",
                "sha256": hashlib.sha256(corpus.encode()).hexdigest(), **(extra or {})}
    if expected and manifest != expected:
        raise ValueError("Replayed corpus or source manifest differs")
    if output.exists() and output.read_text(encoding="utf-8") != corpus:
        raise ValueError("Refusing to replace a different corpus")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(corpus, encoding="utf-8")
    output.with_suffix(".sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def select_topic(available, previous):
    available = sorted(available, key=lambda r: hashlib.sha256(("tr-wiki-articles-v1:" + r["source"]["title"]).encode()).hexdigest())
    selected, seen = [], set(previous)
    for split, count in (("val", 1), ("test", 1), ("train", 18)):
        chosen = []
        for r in available:
            revision = r["source"]["revision"]
            if revision in seen or split not in r["eligible_pools"]:
                continue
            chosen.append(r)
            seen.add(revision)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            return None
        selected.extend((r["text"], {**r["source"], "topic": r["topic"], "split": split}) for r in chosen)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--catalog", type=Path)
    selection.add_argument("--manifest", type=Path)
    parser.add_argument("--font-manifest", type=Path, help="Required when selecting new articles")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error("Use one or two workers to limit Wikipedia request concurrency")
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        records = []
        for p in manifest["pages"]:
            text, source = cached_fetch(p, args.cache)
            records.append((text, {**source, "topic": p["topic"], "split": p["split"]}))
        extra = {k: v for k, v in manifest.items() if k in ("selection", "selection_audit")}
        write_corpus(args.output, records, extra, expected=manifest)
    else:
        if not args.font_manifest:
            parser.error("--font-manifest is required with --catalog")
        import generate_tr_dataset as d
        fonts = d.base.load_font_manifest(args.font_manifest)
        common = set.intersection(*(f.codepoints for f in fonts))
        catalog = json.loads(args.catalog.read_text(encoding="utf-8"))

        def prepare(candidate):
            topic, title = candidate
            try:
                text, source = cached_fetch({"title": title}, args.cache)
            except RuntimeError as error:
                # Missing/disambiguation titles can be recorded and skipped;
                # network/HTTP and checksum failures remain fatal.
                return {"topic": topic, "requested_title": title, "error": str(error)}
            except ValueError as error:
                if str(error).startswith("No paragraphs for "):
                    return {"topic": topic, "requested_title": title, "error": str(error)}
                raise
            valid = eligibility(text, common)
            return {"topic": topic, "requested_title": title, "source": source,
                    "text": text, "eligible_pools": valid, "characters": len(text)}

        results, selected, seen = [], [], set()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for topic, titles in catalog["topics"].items():
                available, chosen = [], None
                for start in range(0, len(titles), args.workers):
                    for result in pool.map(prepare, [(topic, t) for t in titles[start:start + args.workers]]):
                        results.append(result)
                        if "source" in result:
                            available.append(result)
                    if len(available) >= 20:
                        chosen = select_topic(available, seen)
                        if chosen:
                            break
                if chosen is None:
                    diagnostic = {s: sum(s in r["eligible_pools"] for r in available) for s in ("train", "val", "test")}
                    raise ValueError(f"Need more usable articles for {topic}: {diagnostic}")
                selected.extend(chosen)
                seen.update(s["revision"] for _, s in chosen)
                print(f"Selected {len(selected)}/400 articles; {topic}: checked {len(available)} candidates", flush=True)
        audit = [{k: v for k, v in r.items() if k != "text"} for r in results]
        extra = {"selection": {"catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
                               "topic_count": len(catalog["topics"]), "articles_per_topic": 20,
                               "articles_per_topic_by_split": {"train": 18, "val": 1, "test": 1},
                               "font_sha256": sorted(f.sha256 for f in fonts)},
                 "selection_audit": audit}
        manifest = write_corpus(args.output, selected, extra)
    print(json.dumps({"articles": len(manifest["pages"]), "sha256": manifest["sha256"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
