#!/usr/bin/env python3
"""Check local assets, page links, and HTML anchors in a built MkDocs site.

MkDocs cannot validate URLs embedded in raw HTML.  This catches the common
failure mode where a relative image link works in the Markdown source but is
resolved against a directory URL such as ``design/motivation/`` after deploy.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


ATTRIBUTES = ("src", "href")


class AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in ATTRIBUTES and value:
                self.urls.append(value)
            if value and (name == "id" or (tag == "a" and name == "name")):
                self.anchors.add(value)


def is_local_link(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc:
        return False
    return True


def page_base(site_dir: Path, page: Path, site_prefix: str = "") -> str:
    """Return the browser base URL for a generated HTML page."""
    relative = page.relative_to(site_dir)
    prefix = "/" + site_prefix.strip("/") + "/" if site_prefix.strip("/") else "/"
    if relative.name != "index.html":
        return prefix + relative.as_posix()
    parent = relative.parent.as_posix()
    return prefix if parent == "." else f"{prefix}{parent}/"


def site_path(url: str, site_prefix: str) -> str:
    """Convert a deployed URL path into a path relative to ``site_dir``."""
    path = urlsplit(url).path
    prefix = "/" + site_prefix.strip("/") + "/" if site_prefix.strip("/") else "/"
    if prefix != "/" and path.startswith(prefix):
        path = path[len(prefix) :]
    return unquote(path.lstrip("/"))


def check_site(site_dir: Path, site_prefix: str = "") -> tuple[int, list[str]]:
    """Resolve links as a browser does, then check files and page anchors."""
    pages: dict[Path, AssetParser] = {}
    for page in sorted(site_dir.rglob("*.html")):
        parsed = AssetParser()
        parsed.feed(page.read_text(encoding="utf-8"))
        pages[page] = parsed

    checked = 0
    failures: list[str] = []
    for page, parsed in pages.items():
        for url in sorted(set(parsed.urls)):
            if not is_local_link(url):
                continue
            checked += 1
            resolved = urljoin(page_base(site_dir, page, site_prefix), url)
            label = f"{page.relative_to(site_dir)}: {url}"
            prefix = "/" + site_prefix.strip("/") + "/" if site_prefix.strip("/") else "/"
            if not urlsplit(resolved).path.startswith(prefix):
                failures.append(f"outside site prefix: {label} -> {resolved}")
                continue
            target = site_dir / site_path(resolved, site_prefix)
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                failures.append(f"missing target: {label} -> {target.relative_to(site_dir)}")
                continue
            fragment = unquote(urlsplit(resolved).fragment)
            if fragment and target.suffix.lower() == ".html":
                target_parser = pages.get(target)
                if target_parser is None or fragment not in target_parser.anchors:
                    failures.append(f"missing anchor: {label} -> {target.relative_to(site_dir)}#{fragment}")
    return checked, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("site_dir", type=Path)
    parser.add_argument(
        "--site-prefix",
        default="",
        help="GitHub Pages path prefix, for example /ascend-fft/",
    )
    args = parser.parse_args()
    site_dir = args.site_dir.resolve()
    if not site_dir.is_dir():
        parser.error(f"site directory does not exist: {site_dir}")

    checked, failures = check_site(site_dir, args.site_prefix)
    if failures:
        for failure in failures:
            print(failure)
        return 1
    print(f"checked {checked} local asset/page/anchor links; all targets exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
