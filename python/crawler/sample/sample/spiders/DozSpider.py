# -*- coding: utf-8 -*-

import scrapy

class DmozSpider(scrapy.Spider):
    name = "dmoz" # crawling 작업의 이름
    allowed_domains = ["dmoz.org"] # 사이트의 도메인 
    start_urls = [ # 크롤링할 페이지
        "http://www.dmoz.org/Computers/Programming/Languages/Python/Books/",
        "http://www.dmoz.org/Computers/Programming/Languages/Python/Resources/"
    ]

    def parse(self, response):
        filename = response.url.split("/")[-2]
        with open(filename, 'wb') as f:
            f.write(response.body)