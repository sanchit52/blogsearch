# import scrapy
# import json
# import os
# from urllib.parse import urlparse, urljoin

# class BlogSpider(scrapy.Spider):
#     name = "blog_spider"

#     def start_requests(self):
#         file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'personal_blogs.jsonl'))
#         with open(file_path, encoding='utf-8') as f:
#             for line in f:
#                 url = json.loads(line).get('url')
#                 if url:
#                     yield scrapy.Request(url, callback=self.parse_links, meta={'root': url})

#     def parse_links(self, response):
#         root = response.meta['root']
#         base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(root))
#         links = set(response.css('a::attr(href)').getall())

#         for href in links:
#             full_url = urljoin(base, href)
#             parsed = urlparse(full_url)
#             path = parsed.path.lower()

#             # Filter out feeds, home, category, etc.
#             if any(bad in path for bad in ['/feed', '/rss', '/category', '/tag', '/archive']):
#                 continue

#             # Require path depth >= 2 (e.g., /2024/06/post-title or /blog/post)
#             if len([seg for seg in parsed.path.split('/') if seg]) < 2:
#                 continue

#             # Don't go to other domains
#             if parsed.netloc != urlparse(root).netloc:
#                 continue

#             yield scrapy.Request(full_url, callback=self.parse_post, meta={'origin': root})

#     def parse_post(self, response):
#         # Look for <article> or long text content
#         paragraphs = response.css('article p::text').getall()
#         if not paragraphs:
#             paragraphs = response.css('p::text').getall()

#         content = '\n\n'.join(p.strip() for p in paragraphs if p.strip())

#         if len(content.split()) < 100:  # discard very short pages
#             return

#         title = response.css('title::text').get(default='').strip()
#         yield {
#             'url': response.url,
#             'origin': response.meta['origin'],
#             'title': title,
#             'content': content
#         }
import scrapy

class TestSpider(scrapy.Spider):
    name = "test_spider"
    start_urls = ['https://example.com']

    def parse(self, response):
        yield {
            "title": response.css("h1::text").get(),
            "url": response.url
        }
