import requests
from bs4 import BeautifulSoup
import time

def extract_main_text(url):
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title else ""
        paras = soup.find_all("p")
        text = "\n".join(p.get_text() for p in paras if len(p.get_text()) > 30)
        return title, text[:2000]  # Limit size
    except Exception as e:
        return None, None

def crawl_links(input_file="crawler/blog_links.txt"):
    with open(input_file, "r") as f:
        urls = [line.strip() for line in f.readlines()]

    results = []
    for url in urls[:30]:  # Start small to avoid IP ban
        title, text = extract_main_text(url)
        if title and text:
            results.append({"title": title, "url": url, "content": text})
            print(f"✅ {title}")
        time.sleep(1)

    return results

if __name__ == "__main__":
    results = crawl_links()
    import json
    with open("crawler/blog_corpus.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
