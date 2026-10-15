#!/usr/bin/env python3
"""Download Turkish Wikipedia paragraphs; replay exact revisions from a manifest."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import unicodedata
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://tr.wikipedia.org/w/api.php"
TITLES = ["Türkiye", "Türkçe", "İstanbul", "Ankara", "İzmir", "Bilim", "Müzik", "Edebiyat"]


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.skip = 0
        self.parts = []
        self.lines = []

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.active, self.parts = True, []
        if self.active and tag == "sup":
            self.skip += 1

    def handle_endtag(self, tag):
        if tag == "sup" and self.skip:
            self.skip -= 1
        if tag == "p" and self.active:
            line = " ".join(unicodedata.normalize("NFC", "".join(self.parts)).split())
            if line:
                self.lines.append(line)
            self.active, self.skip = False, 0

    def handle_data(self, data):
        if self.active and not self.skip:
            self.parts.append(data)


def extract(html):
    parser = Paragraphs()
    parser.feed(html)
    # TextReader skips newlines; retain a space between paragraphs.
    return " \n".join(parser.lines)


def fetch(page):
    params = {"action": "parse", "format": "json", "prop": "text|revid", "redirects": 1}
    params.update({"oldid": page["revision"]} if "revision" in page else {"page": page["title"]})
    request = Request(API + "?" + urlencode(params), headers={"User-Agent": "SynthDoG-Turkish/1.0 (corpus preparation)"})
    with urlopen(request, timeout=60) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    parsed = result["parse"]
    text = extract(parsed["text"]["*"])
    if not text:
        raise ValueError("No paragraphs for " + parsed["title"])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if page.get("sha256", digest) != digest:
        raise ValueError("Revision text changed for " + parsed["title"])
    return text, {"title": parsed["title"], "revision": parsed["revid"],
                  "url": "https://tr.wikipedia.org/w/index.php?oldid=" + str(parsed["revid"]),
                  "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("resources/corpus/trwiki.txt"))
    parser.add_argument("--manifest", type=Path, help="Replay revisions and verify hashes from a previous .sources.json")
    parser.add_argument("--titles", nargs="+", default=TITLES)
    args = parser.parse_args()
    pages = json.loads(args.manifest.read_text(encoding="utf-8"))["pages"] if args.manifest else [{"title": title} for title in args.titles]
    texts, sources = [], []
    for page in pages:
        text, source = fetch(page)
        texts.append(text)
        sources.append(source)
    corpus = " \n".join(texts) + " \n"
    manifest = {"language": "tr", "source": API, "pages": sources,
                "license": "CC-BY-SA-4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "attribution": "Turkish Wikipedia contributors; see each revision URL and its page history.",
                "transformation": "HTML paragraphs, references removed, NFC normalization, whitespace collapsed",
                "sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest()}
    if args.manifest and json.loads(args.manifest.read_text(encoding="utf-8"))["sha256"] != manifest["sha256"]:
        raise ValueError("Corpus hash does not match replay manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corpus, encoding="utf-8")
    args.output.with_suffix(".sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(corpus)} characters from {len(sources)} pages to {args.output}")


if __name__ == "__main__":
    main()
