# blogcrawler/items.py
import scrapy

class BlogPost(scrapy.Item):
    title = scrapy.Field()
    url = scrapy.Field()
    content = scrapy.Field()
    label = scrapy.Field()
