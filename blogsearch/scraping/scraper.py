# File: blogsearch/scraping/scraper.py

import os
import json
import time
import requests
from newspaper import Article
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

# ——— Paths ———
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(HERE, "sources")
RAW_DIR    = os.path.join(HERE, "..", "data", "raw")

# Source file names must match what's on disk:
PERSONAL_SRC       = os.path.join(SOURCE_DIR, "personal_sources.txt")
NON_PERSONAL_SRC   = os.path.join(SOURCE_DIR, "non_personal_sources.txt")

# Ensure output folder exists
os.makedirs(RAW_DIR, exist_ok=True)

# ——— Load seed URLs ———
with open(PERSONAL_SRC, "r", encoding="utf-8") as f:
    personal_seeds = [u.strip() for u in f if u.strip()]

with open(NON_PERSONAL_SRC, "r", encoding="utf-8") as f:
    nonpersonal_seeds = [u.strip() for u in f if u.strip()]

HEADERS = {"User-Agent": "Mozilla/5.0"}

def is_blog_post(url):
    """Heuristic: looks for likely blog‐style URLs."""
    checks = [
        "202", "blog", "post", "article",
        "journey", "career", "story", "experience",
        "tech", "dev", "manager", "project"
    ]
    u = url.lower()
    return any(c in u for c in checks) and not u.endswith(".xml")

def filter_blog(article):
    """Drop posts that are too short or too diary‐like."""
    text = (article.text or "").strip()
    title = (article.title or "").strip()
    if len(text.split()) < 200:
        return False
    bad_kw = ["dream log", "dear diary", "my dreams", "astrological chart"]
    if any(kw in text.lower() for kw in bad_kw):
        return False
    bad_title = ["entry", "daily", "log", "reflection"]
    if any(bt in title.lower() for bt in bad_title):
        return False
    return True

def extract_links(homepage):
    """Crawl one page and return internal links that look like posts."""
    try:
        r = requests.get(homepage, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        base = homepage
        links = set()
        for a in soup.find_all("a", href=True):
            full = urljoin(base, a["href"])
            if is_blog_post(full) and full.startswith(base):
                links.add(full)
        return list(links)
    except Exception:
        return []

def scrape_article(url):
    """Download & parse via newspaper3k, then filter."""
    try:
        art = Article(url)
        art.download(); art.parse()
        if not filter_blog(art):
            return None
        return {"title": art.title, "url": url, "content": art.text}
    except Exception:
        return None

def crawl_and_scrape(seeds, label, out_fname):
    """For each seed homepage: crawl links, scrape articles, save JSON."""
    out_path = os.path.join(RAW_DIR, out_fname)
    seen = set()
    results = []

    for home in tqdm(seeds, desc=f"→ {label} seeds"):
        links = extract_links(home)
        for link in links:
            if link in seen:
                continue
            seen.add(link)
            art = scrape_article(link)
            if art:
                art["label"] = label
                results.append(art)
            time.sleep(0.3)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {len(results)} “{label}” articles → {out_path}")

if __name__ == "__main__":
    crawl_and_scrape(personal_seeds,     "personal",     "personal_blogs.json")
    crawl_and_scrape(nonpersonal_seeds,  "non_personal", "non_personal_blogs.json")
