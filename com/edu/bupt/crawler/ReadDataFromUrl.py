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

    # load data from disk by normal file utils
    def getRecord(self, filename, fileUtils, row, col):
        records = fileUtils.read_csv(filename)
        return records[row][col]

    # load data from disk by csv reader
    def getRecordByCSV(self, filename, fileUtils, row, col):
        records = fileUtils.read(filename)
        items = zip(item.strip().split(' ') for item in records.split('\n'))
        print items[0][row][col]

if __name__ == '__main__':
    me = Crawler()
    url = "http://www.stats202.com/stats202log.txt"
    data = me.read(url)
    filename = 'C:\Users\Administrator\Desktop\exp.csv'
    fileUtils = fu.FileUtils()
    # save the url contents to disk
    fileUtils.save(filename, data)
    # get the 1 row and 1 col value from csv file
    print me.getRecord(filename, fileUtils, 0, 0)
    # another method to get 1 row and 1 col value from csv file
    print me.getRecordByCSV(filename, fileUtils, 0, 0)

