# -*- coding:utf8 -*-
import csv

import decimal

ctx = decimal.Context()
ctx.prec = 5

def float_to_str(f):
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def transfer(source_path, dest_path):
    res = []
    with open(source_path, 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            res.append([row[0], row[1], float_to_str(float(row[2][:-1]))])

    with open(dest_path, 'w') as f:
        f_csv = csv.writer(f)
        for line in res:
            f_csv.writerow(line)

if __name__ == '__main__':
    source = r"./pig_result/pig_6000_id_testB.csv"
    dest = r"new_pig_testB.csv"
    transfer(source, dest)
