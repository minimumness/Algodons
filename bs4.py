import bs4  # BeautifulSoup
import requests
from datetime import datetime


def get_reuters_article_body(URL):
    
    page = requests.get(URL)
    soup = bs4.BeautifulSoup(page.text)
    element = soup.select('div.StandardArticleBody_body')
    body = element[0].get_text()
    return body

def convert_date_to_epoch(time):
    time = time[:17]
    obj = datetime.strptime(time, '%Y%m%d %I:%M %p')
    epoch = obj.timestamp()
    return epoch
    
a = get_reuters_article_body('http://www.reuters.com/article/entertainmentNews/idUSN2820662920070102')
b = convert_date_to_epoch('20070103 11:57 PM EST')