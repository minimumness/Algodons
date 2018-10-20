import bs4  # BeautifulSoup
import requests

results = []

page = requests.get('http://www.reuters.com/article/entertainmentNews/idUSN2820662920070102')
soup = bs4.BeautifulSoup(page.text)

element = soup.select('div.StandardArticleBody_body')
movie = element[0].get_text()

results.append(movie)