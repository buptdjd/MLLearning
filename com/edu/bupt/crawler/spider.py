# -*- coding: UTF-8 -*-
__author__ = 'Administrator'


import httplib
import urllib
import json
import time
from bs4 import BeautifulSoup

'''
    this spider is used to fetch the information of zhihu user's followees
'''
class Spider:
    '''
        hash_id is the zhihu user's uniform id
        xsrf
        url is www.zhihu.com
        headers is the http request parameters
    '''
    def __init__(self, hash_id, xsrf, url, headers):
        self.hash_id = hash_id
        self.xsrf = xsrf
        self.url = url
        self.headers = headers

    '''
        get zhihu user followees list
    '''

    def get_followees_user_home_url(self, offset):
        connection = httplib.HTTPSConnection(self.url)
        params = {
            'offset': offset,
            'order_by': 'created',
            'hash_id': self.hash_id
        }

        data = urllib.urlencode({
            'method': 'next',
            'params': json.dumps(params),
            '_xsrf': self.xsrf
        }).encode("utf-8")

        connection.request('POST', '/node/ProfileFolloweesListV2', data, self.headers)

        response = connection.getresponse().read().decode("utf-8")

        print response
        data = json.loads(response)

        urls = []
        for html in data['msg']:
            soup = BeautifulSoup(html, "html.parser")
            urls.append(soup.find("a")['href'])

        return urls

    '''
        with the urls to get the information of zhihu users
    '''
    def get_user(self, uri):
        connection = httplib.HTTPSConnection(self.url)

        connection.request('GET', uri + "/about", None, self.headers)

        response = connection.getresponse().read().decode('utf-8')
        soup = BeautifulSoup(response, "html.parser")

        # gender
        if len(soup.select('.gender')) > 0:
            if soup.select('.gender')[0].i['class'][1] == 'icon-profile-male':
                gender = "male"
            else:
                gender = "female"
        else:
            gender = "unknown"

        if len(soup.select('.business')) > 0:
            business = soup.select('.business')[0].string
        else:
            business = ""

        if len(soup.select('.location')) > 0:
            location = soup.select('.location')[0].string
        else:
            location = ""

        user = {
            'nickname': soup.select('.title-section')[0].find('a').string,
            'img': soup.select('.zm-profile-header-avatar-container ')[0].find('img')['srcset'],
            'location': location,
            'business': business,
            'gender': gender,
        }
        return user


if __name__ == '__main__':
    print("begin to fetch the followees, the amount of users is 100")
    headers = {
            'Content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'X-Requested-With': 'XMLHttpRequest',
            'Cookie': '_za=0e976da7-a6ad-43ee-bb37-47185d27ad72; _ga=GA1.2.1725015590.1427104145; q_c1=d1235b20f4154276ae464c182bd6818b|1452516685000|1404982603000; aliyungf_tc=AQAAAHwvvF8L9AwADCj/chD5ZFlFoxjv; _xsrf=a2debc0aae4544e68412db1a03fabdb7; cap_id="YThlNmYzYzFlYTkxNGRmZGEzNGQ0NDE2ODhiZjUwNzg=|1453897141|88e31bea24b2e1692cfb9ddaa8b3455a49bc4f69"; __utmt=1; __utma=51854390.1725015590.1427104145.1453815425.1453895181.4; __utmb=51854390.7.9.1453895214850; __utmc=51854390; __utmz=51854390.1453895181.4.4.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utmv=51854390.000--|3=entry_date=20140710=1; z_c0="QUFBQVhEWWVBQUFYQUFBQVlRSlZUYjFDMEZaZ3IweXlVTy1rakxQNmdsYUQxYW5WZTFyQS1nPT0=|1453897149|ad2c9192f93bf721277e004dd619c561a0f9dea0"; unlock_ticket="QUFBQVhEWWVBQUFYQUFBQVlRSlZUY1c4cUZhTGNwemZWTFI4cHNXUEZvanlGR0RKcG9IWk93PT0=|1453897149|e089a0f0e53d789d63749296dc90f90599c737c6"',
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36'
    }
    urls = []
    users = []
    hash_id = "7407795460968de9aa7f60d890b29c39"
    xsrf = "a2debc0aae4544e68412db1a03fabdb7"
    spider = Spider(hash_id, xsrf, 'www.zhihu.com', headers)
    for n in [0, 20, 40, 60, 80]:
        print("fetching the user %d to %d" % (n, n + 20))
        urls += spider.get_followees_user_home_url(0)
        # protect my spider from shielding by zhihu
        time.sleep(1)

    print("begin to get user information with the followee url...")
    for url in urls:
        users.append(spider.get_user(url))
        # protect my spider from shielding by zhihu
        time.sleep(1)

    for user in users:
        print user['nickname']

    # get_user("/people/luo-yi-28-86")
