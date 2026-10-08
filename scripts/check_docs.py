"""Check mdBook output: assets, internal links, API coverage, and publication scope."""

import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for key in ("href", "src"):
            if key in attrs:
                self.links.append(attrs[key])


def main(directory):
    root = Path(directory).resolve()
    pages = {p: Page(p.read_text()) for p in root.rglob("*.html")}
    assert root / "index.html" in pages, "Missing mdBook homepage"
    for path, page in pages.items():
        assert "<!-- api:" not in path.read_text(), f"Unrendered API comment in {path}"
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or url.path.startswith("/"):
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            assert target.is_file(), f"{path.relative_to(root)}: missing {link}"
            if url.fragment and target in pages:
                assert unquote(url.fragment) in pages[target].ids, (
                    f"{path.relative_to(root)}: missing anchor {link}"
                )
    for page, constructor in (
        ("policies", "Policy"),
        ("execution", "Executor"),
        ("optimizers", "EliteSearch"),
    ):
        content = (root / "api" / f"{page}.html").read_text()
        text = html.unescape(re.sub(r"<[^>]+>", "", content))
        assert f"{constructor}(" in text, constructor
    ids = pages[root / "api/optimizers.html"].ids
    for variant in ("paper", "improved"):
        for member in ("done", "best", "propose", "update"):
            assert f"research.alphaevolve.{variant}.agent.AlphaEvolve.{member}" in ids
    for example in ("existing", "custom"):
        content = (root / "examples" / f"{example}-system.html").read_text()
        assert "async def main" in html.unescape(re.sub(r"<[^>]+>", "", content))
        assert "{{#include" not in content and "--8&lt;--" not in content
    search_files = list(root.glob("searchindex-*.js"))
    assert len(search_files) == 1, "Missing or stale mdBook search index"
    urls = re.search(r'"doc_urls":(\[.*?\])', search_files[0].read_text())
    assert urls, "Missing search document URLs"
    for url in json.loads(urls[1]):
        assert url.split("#")[0] == "index.html" or url.startswith(
            ("guide/", "api/", "examples/")
        ), url
    for file in root.rglob("*"):
        assert not any(
            part in {"maintainer", "superpowers", "research"}
            for part in file.relative_to(root).parts
        ), file
    print(f"mdBook: {len(pages)} pages; assets, links, API, examples, and search scope OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "book")
