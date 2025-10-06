#!/usr/bin/env python3
"""
github_pipeline.py
Pipeline complet:
- collecte métadonnées et README via GitHub API (REST + GraphQL)
- sauvegarde JSON brut
- construit pandas DataFrames
- export CSV / Parquet
- pipeline de labeling heuristique pour multiclass
"""

import os, time, json, math, re
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
from tqdm import tqdm
from dateutil import parser as du_parser
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# CONFIG
OUTDIR = Path("gh_data")
OUTDIR.mkdir(exist_ok=True)
RAW_DIR = OUTDIR / "raw"
RAW_DIR.mkdir(exist_ok=True)
CSV_DIR = OUTDIR / "csv"
CSV_DIR.mkdir(exist_ok=True)
PARQUET_DIR = OUTDIR / "parquet"
PARQUET_DIR.mkdir(exist_ok=True)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("Set GITHUB_TOKEN environment variable before running")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "gh-pipeline/1.0"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# safe request with exponential backoff
@retry(wait=wait_exponential(multiplier=1, min=1, max=60), stop=stop_after_attempt(6),
       retry=retry_if_exception_type((requests.exceptions.RequestException,)))
def safe_get(url, params=None):
    r = SESSION.get(url, params=params, timeout=30)
    if r.status_code == 403:
        # maybe rate limited -> inspect headers
        reset = r.headers.get("X-RateLimit-Reset")
        remaining = r.headers.get("X-RateLimit-Remaining")
        if reset:
            wait_seconds = max(int(reset) - int(time.time()), 0) + 5
            print(f"[RATE LIMIT] sleeping {wait_seconds}s (remaining={remaining})")
            time.sleep(wait_seconds)
            raise requests.exceptions.RequestException("Retry after rate limit reset")
    r.raise_for_status()
    return r

def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -------------- collectors --------------
def fetch_repo_by_fullname(full_name: str) -> Dict:
    """Get repo metadata (REST). full_name = owner/repo"""
    url = f"https://api.github.com/repos/{full_name}"
    r = safe_get(url)
    return r.json()

def fetch_readme(full_name: str) -> Optional[Dict]:
    """Get README (raw + decoded via API)"""
    url = f"https://api.github.com/repos/{full_name}/readme"
    r = safe_get(url)
    if r.status_code == 200:
        return r.json()
    return None

def fetch_contributors_count(full_name: str, per_page=100) -> int:
    # contributors endpoint supports anon param; we just count pages until empty
    url = f"https://api.github.com/repos/{full_name}/contributors"
    params = {"per_page": per_page, "anon": "true", "page": 1}
    total = 0
    while True:
        r = safe_get(url, params=params)
        arr = r.json()
        total += len(arr)
        if len(arr) < per_page:
            break
        params["page"] += 1
    return total

def fetch_prs_count(full_name: str, state="all") -> int:
    url = f"https://api.github.com/repos/{full_name}/pulls"
    params = {"state": state, "per_page": 1}
    r = safe_get(url, params=params)
    # GitHub doesn't return total_count in headers for pulls; use search API for counts:
    owner, repo = full_name.split("/")
    q = f"repo:{owner}/{repo} is:pr"
    return fetch_search_count(q)

def fetch_issues_count(full_name: str) -> int:
    owner, repo = full_name.split("/")
    q = f"repo:{owner}/{repo} is:issue"
    return fetch_search_count(q)

def fetch_releases_count(full_name: str) -> int:
    url = f"https://api.github.com/repos/{full_name}/releases"
    params = {"per_page": 1}
    r = safe_get(url, params=params)
    # Link header method: but releases endpoint returns [] if none; count by paging cheaply:
    # We try to read total from "Link" header if present; otherwise count pages up to a cap
    link = r.headers.get("Link")
    if not link:
        arr = r.json()
        return len(arr)
    # if Link header exists, we attempt to parse last page number
    m = re.search(r'&page=(\d+)>; rel="last"', link)
    if m:
        return int(m.group(1))
    return 0

def fetch_commits_count(full_name: str) -> int:
    owner, repo = full_name.split("/")
    q = f"repo:{owner}/{repo} is:commit"
    return fetch_search_count(q)

def fetch_search_count(query: str) -> int:
    # Use the search API which returns total_count
    url = "https://api.github.com/search/issues"  # for issues/prs search
    # For commits, use commit search: /search/commits requires a special accept header
    # We'll adapt based on query content
    if "is:commit" in query:
        url = "https://api.github.com/search/commits"
        headers = SESSION.headers.copy()
        headers["Accept"] = "application/vnd.github.cloak-preview"  # commit search preview
        r = SESSION.get(url, params={"q": query}, headers=headers)
    elif "is:pr" in query or "is:issue" in query:
        r = safe_get(url, params={"q": query})
    else:
        r = safe_get("https://api.github.com/search/repositories", params={"q": query})
    if r.status_code != 200:
        return 0
    j = r.json()
    return j.get("total_count", 0)

# Helper to fetch topics (requires specific accept header)
def fetch_topics(full_name: str) -> List[str]:
    url = f"https://api.github.com/repos/{full_name}/topics"
    headers = SESSION.headers.copy()
    headers["Accept"] = "application/vnd.github.mercy-preview+json"
    r = SESSION.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get("names", [])
    return []

# Batch fetcher for a list of repo fullnames
def collect_repos(fullnames: List[str], save_raw=True):
    rows = []
    for full in tqdm(fullnames, desc="repos"):
        try:
            repo = fetch_repo_by_fullname(full)
        except Exception as e:
            print(f"Error fetching {full}: {e}")
            continue
        if save_raw:
            save_json(repo, RAW_DIR / f"{full.replace('/', '__')}_repo.json")
        # readme
        readme = None
        try:
            rREADME = fetch_readme(full)
            if rREADME:
                readme = {
                    "download_url": rREADME.get("download_url"),
                    "name": rREADME.get("name"),
                    "encoding": rREADME.get("encoding"),
                    "content": rREADME.get("content")  # base64 encoded
                }
                save_json(rREADME, RAW_DIR / f"{full.replace('/', '__')}_readme.json")
        except Exception:
            readme = None

        # quick counts using search when available
        contributors = None
        try:
            contributors = fetch_contributors_count(full)
        except Exception:
            contributors = None

        try:
            issues_count = fetch_issues_count(full)
            prs_count = fetch_prs_count(full)
        except Exception:
            issues_count = repo.get("open_issues_count", None)
            prs_count = None

        try:
            commits_count = fetch_commits_count(full)
        except Exception:
            commits_count = None

        topics = fetch_topics(full)

        row = {
            "full_name": repo.get("full_name"),
            "name": repo.get("name"),
            "owner": repo.get("owner", {}).get("login"),
            "description": repo.get("description"),
            "created_at": repo.get("created_at"),
            "updated_at": repo.get("updated_at"),
            "pushed_at": repo.get("pushed_at"),
            "language": repo.get("language"),
            "forks_count": repo.get("forks_count"),
            "stargazers_count": repo.get("stargazers_count"),
            "watchers_count": repo.get("watchers_count"),
            "open_issues_count": repo.get("open_issues_count"),
            "topics": topics,
            "has_issues": repo.get("has_issues"),
            "has_projects": repo.get("has_projects"),
            "has_wiki": repo.get("has_wiki"),
            "has_pages": repo.get("has_pages"),
            "license": repo.get("license", {}).get("spdx_id") if repo.get("license") else None,
            "readme": readme,
            "contributors_count": contributors,
            "issues_count_search": issues_count,
            "prs_count_search": prs_count,
            "commits_count_search": commits_count,
            "size_kb": repo.get("size"),
            "archived": repo.get("archived"),
            "disabled": repo.get("disabled"),
            "fork": repo.get("fork"),
            "raw_repo_json": repo
        }
        rows.append(row)
        # small sleep to be nice
        time.sleep(0.1)
    df = pd.DataFrame(rows)
    return df

# -------------- labeling heuristics --------------
CATEGORIES = [
    "web", "network", "selfhost", "low_level", "plugin", "cli_tool",
    "mobile", "data_science", "infrastructure", "documentation", "other"
]

KEYWORDS = {
    "web": ["web", "html", "css", "react", "vue", "angular", "next.js", "nuxt", "frontend"],
    "network": ["network", "socket", "http", "tcp", "udp", "dns", "proxy", "netstat", "p2p"],
    "selfhost": ["selfhost", "self-host", "selfhosted", "self-hosted", "home server", "homelab"],
    "low_level": ["c", "c++", "rust", "embedded", "driver", "kernel", "low-level", "microcontroller"],
    "plugin": ["plugin", "extension", "vscode-extension", "wordpress-plugin", "gimp-plugin", "jenkins"],
    "cli_tool": ["cli", "command-line", "tool", "script", "utility"],
    "mobile": ["android", "ios", "flutter", "kotlin", "swift", "react-native"],
    "data_science": ["data", "pandas", "numpy", "machine learning", "ml", "tensorflow", "pytorch", "notebook"],
    "infrastructure": ["docker", "kubernetes", "helm", "ansible", "terraform", "ci", "cd", "devops"],
    "documentation": ["doc", "docs", "documentation", "readme", "guide", "tutorial"]
}

def decode_base64(content_b64: Optional[str]) -> Optional[str]:
    if not content_b64:
        return None
    import base64
    try:
        b = base64.b64decode(content_b64.encode("utf-8"), validate=True)
        return b.decode("utf-8", errors="replace")
    except Exception:
        return None

def label_repo_auto(row: Dict) -> str:
    """Heuristics: search in topics, name, description, readme content, language."""
    text_blob = " ".join([
        " ".join(row.get("topics") or []),
        row.get("name") or "",
        row.get("description") or ""
    ]).lower()
    readme_text = None
    if row.get("readme") and row["readme"].get("content"):
        readme_text = decode_base64(row["readme"]["content"]).lower()
        text_blob += " " + (readme_text[:10000] if readme_text else "")

    # search for keywords
    scores = {c: 0 for c in CATEGORIES}
    for cat, kws in KEYWORDS.items():
        for kw in kws:
            if kw in text_blob:
                scores[cat] += 1

    # language signal
    lang = (row.get("language") or "").lower()
    if lang in ("c", "c++", "rust"):
        scores["low_level"] += 2
    if lang in ("python", "r", "jupyter notebook", "jupyter"):
        scores["data_science"] += 1
    if lang in ("javascript", "typescript", "html", "css"):
        scores["web"] += 1

    # simple winner-takes-all with threshold
    best = max(scores.items(), key=lambda x: x[1])
    if best[1] == 0:
        return "other"
    return best[0]

# -------------- main flow --------------
def main():
    # 1) Choose repos to collect. Two options:
    # - a) from a file of "owner/repo" lines
    # - b) from a search query (ex: popular repos with README)
    # here we support both. Adjust accordingly.
    seed_file = Path("repos_list.txt")
    if seed_file.exists():
        fullnames = [l.strip() for l in open(seed_file, "r", encoding="utf-8") if l.strip()]
    else:
        # default small seed (exemples) — remplace par ta propre recherche/loader
        fullnames = [
            "torvalds/linux",
            "django/django",
            "pallets/flask",
            "scikit-learn/scikit-learn",
            "facebook/react",
            "vuejs/vue",
            "golang/go",
            "rust-lang/rust",
            "kubernetes/kubernetes"
        ]

    print(f"Collecting {len(fullnames)} repos ...")
    df = collect_repos(fullnames, save_raw=True)

    # decode readmes to a separate column
    df["readme_text"] = df["readme"].apply(lambda r: decode_base64(r["content"]) if r and r.get("content") else None)

    # label automatically
    df["auto_label"] = df.apply(lambda r: label_repo_auto(r.to_dict()), axis=1)

    # enrich: compute README length, stars_per_kb, etc.
    df["readme_len"] = df["readme_text"].apply(lambda s: len(s) if s else 0)
    df["stars_per_kb"] = df.apply(lambda r: (r["stargazers_count"] / max(1, r["size_kb"])) if r["size_kb"] else r["stargazers_count"], axis=1)

    # save dataframes
    df.to_csv(CSV_DIR / "repos_dataset.csv", index=False)
    df.to_parquet(PARQUET_DIR / "repos_dataset.parquet", index=False)

    # Save a small schema / preview
    print("Saved dataset columns:", list(df.columns))
    print("Saved CSV and Parquet in:", CSV_DIR, PARQUET_DIR)

if __name__ == "__main__":
    main()
