import requests
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
from fake_useragent import UserAgent
import requests


ua = UserAgent()
print(ua.chrome)
header = {'User-Agent':str(ua.chrome)}
headers = {
    "content-type": "application/json"
}
print(header)
url='http://httpbin.org/redirect/3'
# r = requests.get('http://httpbin.org/redirect/3')
r = requests.head(url, allow_redirects=True)

for i in range(len(r.history)):
    print(r.history[0].url)
print(r)