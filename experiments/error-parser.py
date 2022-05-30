from ast import arg
import re
import sys
import os

from pyparsing import Regex


# use: <python interpreter> error-parser.py <file in> <file out>
assert(len(sys.argv) == 3)

file_in = sys.argv[1]
file_out = sys.argv[2]

with open(file_in) as f:
    regex = re.compile(r'(\w+.\w+:\d+:\d+)(?:.*)\[CWE-(.*?)\]')
    for line in f.readlines():
        m = regex.findall(line)
        if (not m):
            continue
        path = m[0][0]
        cwe = m[0][1]

        print(path)
        print(cwe)
