import requests
from bs4 import BeautifulSoup as bs
from time import sleep
url = 'http://server.ataxx.org'
html = requests.get(url)
soup = bs(html.text, 'html.parser')
players = [x.get('href') for x in soup.find('table').find_all('a')]
pgns = []
for player in players:
    soup2 = bs(requests.get(url+'/'+player).text, 'html.parser')
    pgns += [tr.find('a').get('href') for tr in soup2.find_all('table')[2].find_all('tr')[1:] if tr.find_all('td')[7].text == '']
print(len(pgns))
for i, link in enumerate(pgns):
    name = link.split('=')[-1]
    f = open('data/'+name+'.txt', 'w')
    with requests.Session() as s:
        f.write(s.get(url+'/'+link).text)
    sleep(0.5)
	if i % 10 == 0:
		print(i)