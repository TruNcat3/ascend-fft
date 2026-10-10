#!/usr/bin/env python3
"""Check local image and downloadable-figure links in a built MkDocs site.

MkDocs cannot validate URLs embedded in raw HTML.  This catches the common
failure mode where a relative image link works in the Markdown source but is
resolved against a directory URL such as ``design/motivation/`` after deploy.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


ASSET_SUFFIXES = (".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf")
ATTRIBUTES = ("src", "href")


class AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in ATTRIBUTES and value:
                self.urls.append(value)


def is_local_asset(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc:
        return False
    if url.startswith(("#", "data:", "javascript:")):
        return False
    return Path(parsed.path.lower()).suffix in ASSET_SUFFIXES


def page_base(site_dir: Path, page: Path) -> str:
    """Return the browser base URL for a generated HTML page."""
    parent = page.relative_to(site_dir).parent.as_posix()
    return "/" if parent == "." else f"/{parent}/"


def site_path(url: str, site_prefix: str) -> str:
    """Convert a deployed URL path into a path relative to ``site_dir``."""
    path = urlsplit(url).path
    prefix = "/" + site_prefix.strip("/") + "/" if site_prefix.strip("/") else "/"
    if prefix != "/" and path.startswith(prefix):
        path = path[len(prefix) :]
    return unquote(path.lstrip("/"))


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

    missing: list[tuple[Path, str, Path]] = []
    for page in sorted(site_dir.rglob("*.html")):
        parser_ = AssetParser()
        parser_.feed(page.read_text(encoding="utf-8"))
        for url in sorted(set(parser_.urls)):
            if not is_local_asset(url):
                continue
            target = site_dir / site_path(urljoin(page_base(site_dir, page), url), args.site_prefix)
            if not target.is_file():
                missing.append((page.relative_to(site_dir), url, target.relative_to(site_dir)))

    if missing:
        for page, url, target in missing:
            print(f"missing asset: {page}: {url} -> {target}")
        return 1

    checked = 0
    for page in site_dir.rglob("*.html"):
        parser_ = AssetParser()
        parser_.feed(page.read_text(encoding="utf-8"))
        checked += sum(is_local_asset(url) for url in set(parser_.urls))
    print(f"checked {checked} local image/document links; all targets exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
