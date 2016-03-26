__author__ = 'Administrator'
import csv


# This module is a file utils
class FileUtils:

    def __init__(self):
        pass

    def save(self, filename, data):
        f = open(filename, "a+")
        f.write(data)
        f.close()

    # use normal file utils to resolve the csv files
    def read(self, filename):
        f = open(filename, "r+")
        data = f.read()
        f.close()
        return data

    # use the csv utils to resolve the csv files
    def read_csv(self, filename):
        contents = csv.reader(file(filename, 'r+'))
        ret = []
        for line in contents:
            ret.append(line[0].split(' '))
        return ret

    # get CSV Reader
    def getCsvReader(self, filename):
        return csv.reader(file(filename, 'rb'))


