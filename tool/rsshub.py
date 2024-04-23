import feedparser
from lxml import etree


def get_content(url):
    # 使用 feedparser 库来解析提供的 URL，通常用于读取 RSS 或 Atom 类型的数据流
    data = feedparser.parse(url)
    print(data)
    for news in data['entries']:
        # 通过 xpath 提取干净的文本内容
        summary = etree.HTML(text=news['summary']).xpath('string(.)')
        print("begin:" + summary)


if __name__ == "__main__":
    # 财联社 RSS
    url = "https://rsshub.app/qianzhan/analyst/column/all"
    get_content(url)
