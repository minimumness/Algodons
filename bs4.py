import bs4  # BeautifulSoup
import requests


def get_article_body(URL):
    
    page = requests.get(URL)
    soup = bs4.BeautifulSoup(page.text)
    
    element = soup.select('div.StandardArticleBody_body')
    body = element[0].get_text()
    
    return body
    
a = get_article_body('http://www.reuters.com/article/entertainmentNews/idUSN2820662920070102')
