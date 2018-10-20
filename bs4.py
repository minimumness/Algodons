import requests
from bs4 import BeautifulSoup

page_link = 'http://www.reuters.com/article/entertainmentNews/idUSN2820662920070102'
page_response = requests.get(page_link, timeout=5)
page_content = BeautifulSoup(page_response.content, "html.parser")

textContent = []
for i in range(0, 20):
    paragraphs = page_content.find_all("p")[i].text
    textContent.append(paragraphs)
    
mydivs = BeautifulSoup.findAll("div", {"class": "StandardArticleBody_body"})