# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import arxiv
import json
import os
import sys
from datetime import datetime, timedelta


class DailyArxivPipeline:
    def __init__(self):
        self.page_size = 100
        # arXiv's API guideline is ~1 request per 3 seconds and they enforce it
        # via 429s on shared IPs (e.g. GitHub Actions). 3.0 avoids retries entirely;
        # 2.0 risks occasional 429s but is faster on average. Keep 3.0 unless you
        # see clean runs with no 429 retries in the logs.
        self.client = arxiv.Client(page_size=self.page_size, delay_seconds=3.0, num_retries=3)

    def process_item(self, item: dict, spider):
        item["pdf"] = f"https://arxiv.org/pdf/{item['id']}"
        item["abs"] = f"https://arxiv.org/abs/{item['id']}"
        search = arxiv.Search(
            id_list=[item["id"]],
        )
        paper = next(self.client.results(search))
        item["authors"] = [a.name for a in paper.authors]
        item["title"] = paper.title
        item["categories"] = paper.categories
        item["comment"] = paper.comment
        item["summary"] = paper.summary
        return item