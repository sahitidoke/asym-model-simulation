import pandas
import numpy
import csv

def load_snp_500():
    f = open('data/prices.csv','r')
    d = csv.reader(f)

    data_matrix = []

    for row in d:
        data_matrix += [row]
    
    return data_matrix


