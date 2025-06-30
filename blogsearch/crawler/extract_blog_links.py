import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def is_external(url):
    parsed = urlparse(url)
    return parsed.netloc and not parsed.netloc.endswith("indieweb.org")

def extract_blog_links(seed_url="https://indieweb.org/blogs"):
    response = requests.get(seed_url)
    soup = BeautifulSoup(response.text, "html.parser")

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") and is_external(href):
            links.add(href)

    return list(links)

if __name__ == "__main__":
    blog_links = extract_blog_links()
    print(f"✅ Found {len(blog_links)} blog links")
    with open("crawler/blog_links.txt", "w") as f:
        for link in blog_links:
            f.write(link + "\n")
