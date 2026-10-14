#!/usr/bin/env python3
"""Measure readable Turkish word forms, excluding numeric/ID/confusable tokens."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import re
import unicodedata

from article_sampling import NATURAL

TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
TRANSLATION = str.maketrans({"I": "ı", "İ": "i", "’": "'", "‘": "'", "ʼ": "'"})


def normal(text):
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", text).translate(TRANSLATION).lower())


def words(text):
    return [t for t in TOKEN.findall(normal(text)) if t.replace("'", "").isalpha()]


def analyze(root):
    natural, readable = Counter(), Counter()
    by_split, natural_split, by_topic = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    characters = defaultdict(Counter)
    labels, count = set(), 0
    with (root / "metadata.jsonl").open(encoding="utf-8") as metadata, (root / "labels.txt").open(encoding="utf-8") as targets:
        for line, target in zip(metadata, targets, strict=True):
            row = json.loads(line)
            name, text = target.rstrip("\n").split("\t", 1)
            if (name, text) != (row["file_name"], row["text"]):
                raise ValueError("Metadata/label mismatch")
            count += 1
            characters[row["split"]].update(text)
            labels.add(" ".join(normal(text).split()))
            kind = row["categories"]["content_type"]
            if kind == "confusables":
                continue
            tokens = words(text)
            if kind == "synthetic_id":
                tokens = [t for t in tokens if t != "tr"]
            readable.update(tokens)
            by_split[row["split"]].update(tokens)
            if kind in NATURAL:
                natural.update(tokens)
                natural_split[row["split"]].update(tokens)
                if row.get("source_article"):
                    by_topic[row["source_article"]["topic"]].update(tokens)
    expected = json.loads((root / "metadata.json").read_text(encoding="utf-8"))["count"]
    if count != expected:
        raise ValueError("Wrong dataset size")
    report = {"dataset": root.name, "images": count,
              "method": "NFC, Turkish lowercase I→ı/İ→i, normalize apostrophes; alphabetic word forms with suffixes; numeric/alphanumeric units, synthetic ID prefix and confusable exercises excluded. Inflections, names and abbreviations count separately; no stemming or dictionary validation.",
              "unique_word_forms_including_readable_fields": len(readable),
              "word_occurrences_including_readable_fields": sum(readable.values()),
              "unique_word_forms_by_split_including_readable_fields": {s: len(c) for s, c in by_split.items()},
              "unique_natural_word_forms": len(natural), "natural_word_occurrences": sum(natural.values()),
              "unique_natural_word_forms_by_split": {s: len(c) for s, c in natural_split.items()},
              "character_counts_by_split": {s: dict(sorted(c.items())) for s, c in characters.items()},
              "all_turkish_letters_present_in_training": set("abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ") <= set(characters["train"]),
              "heldout_characters_absent_in_training": sorted((set(characters["val"]) | set(characters["test"])) - set(characters["train"])),
              "unique_normalized_transcriptions_including_structured": len(labels),
              "natural_word_frequency_bands": {"once": sum(n == 1 for n in natural.values()),
                                               "2_to_4": sum(2 <= n <= 4 for n in natural.values()),
                                               "5_to_9": sum(5 <= n <= 9 for n in natural.values()),
                                               "10_or_more": sum(n >= 10 for n in natural.values())},
              "unique_word_forms_by_topic": {t: len(c) for t, c in sorted(by_topic.items())},
              "heldout_word_coverage": {}, "most_frequent_natural_words": natural.most_common(20)}
    train = set(natural_split["train"])
    for split in ("val", "test"):
        counts = natural_split[split]
        unseen = set(counts) - train
        report["heldout_word_coverage"][split] = {
            "unique_word_forms_unseen_in_train": len(unseen),
            "word_occurrences_unseen_in_train": sum(counts[w] for w in unseen),
            "word_occurrences_seen_in_train_percent": round(100 * sum(n for w, n in counts.items() if w in train) / sum(counts.values()), 3)}
    with (root / "word-frequencies.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["word_form", "occurrences", "train", "val", "test"])
        for word, amount in readable.most_common():
            writer.writerow([word, amount, *(by_split[s][word] for s in ("train", "val", "test"))])
    (root / "vocabulary-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    print(json.dumps(analyze(parser.parse_args().dataset), ensure_ascii=False, indent=2))
