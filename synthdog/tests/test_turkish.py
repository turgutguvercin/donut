import json
from pathlib import Path
import sys
import yaml
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_tr_corpus import extract
from validate_tr import validate_fonts, validate_output
from template import SynthDoG


def test_config_preserves_english_settings():
    english = yaml.safe_load((ROOT / "config_en.yaml").read_text())
    turkish = yaml.safe_load((ROOT / "config_tr.yaml").read_text())
    english["document"]["content"]["text"]["path"] = "resources/corpus/trwiki.txt"
    assert turkish == english


def test_wikipedia_normalization():
    assert extract('<p>Tu\u0308rkçe <b>İĞŞ</b><sup>[1]</sup>  ıöçü</p>') == 'Türkçe İĞŞ ıöçü'
    assert extract('<div>No paragraphs</div>') == ''


def test_fonts():
    validate_fonts(ROOT / "resources/font/en")


def test_utf8_save(tmp_path):
    import numpy as np
    generator = SynthDoG.__new__(SynthDoG)
    generator.splits = ["train"]
    generator.split_indexes = [0]
    label = "çğışöü ÇĞİŞÖÜ ıI"
    generator.save(str(tmp_path), {"image": np.zeros((32, 32, 3)), "label": label, "quality": 90, "roi": []}, 0)
    assert validate_output(tmp_path, 1) == [label]
    assert label.encode('utf-8') in (tmp_path / 'train/metadata.jsonl').read_bytes()


def test_validation_rejects_missing_images(tmp_path):
    with pytest.raises(AssertionError):
        validate_output(tmp_path, 20)


def test_revision_replay_and_hash_verification(monkeypatch):
    import hashlib
    import io
    import prepare_tr_corpus as corpus
    response = {"parse": {"title": "Türkçe", "revid": 123, "text": {"*": "<p>Türkçe ıİğş</p>"}}}
    requests = []
    def fake_open(request, timeout):
        requests.append(request.full_url)
        return io.BytesIO(json.dumps(response).encode('utf-8'))
    monkeypatch.setattr(corpus, "urlopen", fake_open)
    text, source = corpus.fetch({"title": "Türkçe"})
    assert source["sha256"] == hashlib.sha256(text.encode('utf-8')).hexdigest()
    assert corpus.fetch(source) == (text, source)
    assert 'oldid=123' in requests[-1]
    with pytest.raises(ValueError, match="changed"):
        corpus.fetch({**source, "sha256": "wrong"})
