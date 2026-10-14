import re
from urllib.parse import urlparse, urlunparse

LISTING_PATH_RE = re.compile(r"^/a/show/\d+(?:/)?$")


def validate_listing_url(value: str) -> str | None:
    try:
        parsed = urlparse(value)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    host = (parsed.hostname or "").lower().rstrip(".")
    if host not in {"krisha.kz", "www.krisha.kz"}:
        return None
    if not LISTING_PATH_RE.fullmatch(parsed.path):
        return None
    return urlunparse(("https", "krisha.kz", parsed.path.rstrip("/"), "", "", ""))
