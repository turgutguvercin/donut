from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import article_sampling as a
import generate_tr_dataset as d
import prepare_tr_balanced_corpus as corpus
from analyze_tr_vocabulary import words


def sources():
    return [{"title": f"topic{t}-article{n}", "topic": f"topic{t}", "revision": t * 20 + n + 1,
             "split": "train" if n < 18 else "val" if n == 18 else "test", "sha256": "test", "url": "https://tr.wikipedia.org"}
            for t in range(20) for n in range(20)]


def test_topic_article_balance_preserves_every_visual_assignment():
    config = json.loads((d.ROOT / "config_tr_recognition_camera.json").read_text())
    fonts = [SimpleNamespace(family=f"f{i}", style="regular") for i in range(20)]
    plan = d.plan_samples(config, fonts, 100000, 20261014)
    original = copy.deepcopy(plan)
    articles = sources()
    a.assign_articles(plan, articles, 20261014)
    counts = Counter(r["article_index"] for r in plan if r["content_type"] in a.NATURAL)
    assert counts == dict.fromkeys(range(400), 150)
    for row, before in zip(plan, original):
        assert {k: v for k, v in row.items() if k != "article_index"} == before
        assert articles[row["article_index"]]["split"] == row["split"]
    replay = copy.deepcopy(original)
    a.assign_articles(replay, articles, 20261014)
    assert replay == plan


def test_manifest_boundaries_hashes_and_exact_replay(tmp_path):
    output = tmp_path / "trwiki.txt"
    records = [("Çiğdem ılık ışıkta üzüm gördü.", {"title": "Bir", "revision": 1, "topic": "a", "split": "train"}),
               ("İşçi küçük çayıra yürüdü.", {"title": "İki", "revision": 2, "topic": "b", "split": "test"})]
    records = [(t, {**s, "sha256": hashlib.sha256(t.encode()).hexdigest()}) for t, s in records]
    manifest = corpus.write_corpus(output, records)
    assert [p["text"] for p in a.load_articles(output)] == [t for t, _ in records]
    assert corpus.write_corpus(output, records, expected=manifest) == manifest
    output.write_text(output.read_text() + "unattributed")
    with pytest.raises(ValueError, match="checksum"):
        a.load_articles(output)


def test_source_audit_rejects_heldout_context_and_unbalanced_articles():
    pages = [p for p in sources() if p["topic"] == "topic0"]
    rows = [{"categories": {"content_type": "short_line"}, "split": p["split"], "source_article": dict(p), "context_article": dict(p)} for p in pages]
    report = a.audit_sources(rows, {"pages": pages})
    assert report["articles_used"] == 20 and report["article_disjoint_splits_including_context"]
    changed = copy.deepcopy(rows)
    changed[0]["context_article"] = pages[-1]
    with pytest.raises(ValueError, match="split mismatch"):
        a.audit_sources(changed, {"pages": pages})
    with pytest.raises(ValueError, match="Unbalanced article"):
        a.audit_sources(rows + [rows[0]] * 2, {"pages": pages})


def test_word_count_turkish_case_suffixes_and_alphanumeric_units():
    assert words("İŞÇİ işçi IŞIK ışık Ankara’nın 1945'te AB123 TR-2026-123456") == ["işçi", "işçi", "ışık", "ışık", "ankara'nın", "tr"]


def test_corrupted_article_cache_is_rejected(tmp_path):
    title = "Türkçe"
    path = tmp_path / (hashlib.sha256(title.encode()).hexdigest() + ".json")
    path.write_text(json.dumps({"text": "tampered", "source": {"sha256": "wrong"}}))
    with pytest.raises(ValueError, match="checksum"):
        corpus.cached_fetch({"title": title}, tmp_path)


def test_article_selection_reserves_heldout_and_never_reuses_revisions():
    available = [{"source": {"title": f"a{i}", "revision": i}, "text": "Türkçe", "topic": "one",
                  "eligible_pools": {s: {} for s in ("train", "val", "test")}} for i in range(21)]
    selected = corpus.select_topic(available, {0})
    assert len(selected) == 20
    assert Counter(p["split"] for _, p in selected) == {"train": 18, "val": 1, "test": 1}
    assert {p["revision"] for _, p in selected} == set(range(1, 21))
    assert corpus.select_topic(available[:19], set()) is None


def test_replay_manifest_rejects_changed_split(tmp_path):
    text = "İşçi çayıra yürüdü."
    source = {"title": "One", "revision": 1, "topic": "a", "split": "train",
              "sha256": hashlib.sha256(text.encode()).hexdigest()}
    path = tmp_path / "corpus.txt"
    manifest = corpus.write_corpus(path, [(text, source)])
    with pytest.raises(ValueError, match="differs"):
        corpus.write_corpus(path, [(text, {**source, "split": "test"})], expected=manifest)


def test_revision_replay_reuses_matching_title_cache(tmp_path, monkeypatch):
    title, text = "Türkçe", "Çiğdem üzüm aldı."
    source = {"title": title, "revision": 42, "sha256": hashlib.sha256(text.encode()).hexdigest()}
    path = tmp_path / (hashlib.sha256(title.encode()).hexdigest() + ".json")
    path.write_text(json.dumps({"text": text, "source": source}))
    def no_request(page):
        raise AssertionError("Verified exact revision should reuse the cache")
    monkeypatch.setattr(corpus, "fetch", no_request)
    assert corpus.cached_fetch(source, tmp_path) == (text, source)
