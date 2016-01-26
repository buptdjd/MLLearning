__author__ = 'Administrator'

import urllib2
import com.edu.bupt.utils.FileUtils as fu


# This module uses Crawler to extract data from web pages
class Crawler:
    def __init__(self):
        pass

    def read(self, url):
        page = urllib2.urlopen(url, timeout=10)
        data = page.read()
        return data

if __name__ == '__main__':
    me = Crawler()
    url = "http://www.stats202.com/stats202log.txt"
    data = me.read(url)
    filename = 'C:\Users\Administrator\Desktop\exp.csv'
    fileUtils = fu.FileUtils()
    # save the url contents to disk
    fileUtils.save(filename, data)

    # get the 1 row and 1 col value from csv file
    # data from disk csv
    data = fileUtils.read_csv(filename)
    print data[0][0]

    # another method to get 1 row and 1 col value from csv file
    # data from disk
    data = fileUtils.read(filename)
    l = data.split('\n')
    records = zip(item.strip().split(' ') for item in l)
    print records[0][0][0]

