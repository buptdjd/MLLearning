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

    def read(self, filename):
        f = open(filename, "r+")
        data = f.read()
        return data

    def read_csv(self, filename):
        contents = csv.reader(file(filename, 'r+'))
        ret = []
        for line in contents:
            ret.append(line[0].split(' '))
        return ret



