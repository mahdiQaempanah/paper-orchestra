"""Look up candidate works in OpenAlex when Semantic Scholar has no match."""

import requests
from thefuzz import fuzz


def openalex_title_search(title, year_hint=None):
    """Return metadata in the shape expected by the literature agent."""
    response = requests.get(
        "https://api.openalex.org/works",
        params={"search": title, "per-page": 5},
        timeout=10,
    )
    response.raise_for_status()
    candidates = response.json().get("results", [])
    matches = []
    for work in candidates:
        candidate_title = work.get("display_name") or ""
        score = fuzz.ratio(title.casefold(), candidate_title.casefold())
        if score >= 90:
            matches.append((score, work.get("publication_year") == year_hint, work))
    if not matches:
        return None

    work = max(matches, key=lambda item: (item[0], item[1]))[2]
    abstract_index = work.get("abstract_inverted_index") or {}
    words = sorted(
        (position, word)
        for word, positions in abstract_index.items()
        for position in positions
    )
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    source_name = source.get("display_name") or ""
    biblio = work.get("biblio") or {}
    first_page = biblio.get("first_page")
    last_page = biblio.get("last_page")
    pages = "-".join(page for page in (first_page, last_page) if page)
    journal = None
    if work.get("type") == "article" and source.get("type") == "journal":
        journal = {"name": source_name, "volume": biblio.get("volume"), "pages": pages}

    return {
        "title": work.get("display_name"),
        "year": work.get("publication_year"),
        "publicationDate": work.get("publication_date"),
        "authors": [
            {"name": author["author"]["display_name"]}
            for author in work.get("authorships", [])
            if (author.get("author") or {}).get("display_name")
        ],
        "abstract": " ".join(word for _, word in words),
        "citationCount": work.get("cited_by_count") or 0,
        "venue": source_name,
        "journal": journal,
        "bibtex_type": "article" if journal else "misc",
        "doi": (work.get("doi") or "").removeprefix("https://doi.org/"),
    }
