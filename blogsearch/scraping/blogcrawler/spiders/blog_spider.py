import scrapy
from pathlib import Path
from newspaper import Article
from blogcrawler.items import BlogPost


class BlogSpider(scrapy.Spider):
    name = "blog_spider"
    custom_settings = {
        "DOWNLOAD_DELAY": 0.5,
    }

    def start_requests(self):
        # Go up two levels from blog_spider.py to reach the scraping/ folder
        base_path = Path(__file__).resolve().parents[2]  # blogsearch/scraping/
        sources_dir = base_path / "sources"

        for label, fname in [
            ("personal", sources_dir / "personal_sources.txt"),
            ("non_personal", sources_dir / "non_personal_sources.txt"),
        ]:
            if not fname.exists():
                self.logger.warning(f"[WARN] Missing source file: {fname}")
                continue

            with open(fname, encoding="utf-8") as f:
                for url in f:
                    url = url.strip()
                    if url:
                        yield scrapy.Request(url=url, callback=self.parse_home, meta={"label": label})


    def parse_home(self, response):
        label = response.meta["label"]
        base_url = response.url

        # Collect all blog post links from homepage
        for link in response.css("a::attr(href)").getall():
            full_url = response.urljoin(link)
            if self.is_blog_post(full_url):
                yield scrapy.Request(full_url, callback=self.parse_post, meta={"label": label})

    def parse_post(self, response):
        label = response.meta["label"]

        try:
            art = Article(response.url)
            art.download()
            art.parse()
        except Exception as e:
            self.logger.warning(f"[WARN] Failed to parse article: {response.url} | {e}")
            return

        if self.filter_blog(art):
            yield BlogPost(
                title=art.title,
                url=response.url,
                content=art.text,
                label=label,
            )

    def is_blog_post(self, url):
        url = url.lower()
        indicators = [
            "202", "blog", "post", "journey", "career", "story",
            "tech", "developer", "engineer", "manager", "experience",
            "project", "my-journey", "how-i", "learning"
        ]
        return any(keyword in url for keyword in indicators) and not url.endswith(".xml")

    def filter_blog(self, article):
        text = (article.text or "").strip()
        title = (article.title or "").strip()

        # Apply basic filters
        if len(text.split()) < 200:
            return False

        bad_keywords = ["dear diary", "dream log", "astrological chart"]
        bad_titles = ["entry", "daily", "log", "reflection"]

        if any(kw in text.lower() for kw in bad_keywords):
            return False
        if any(bt in title.lower() for bt in bad_titles):
            return False

        return True