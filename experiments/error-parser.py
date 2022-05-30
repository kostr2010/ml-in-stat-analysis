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
    regex = re.compile(r'(\w+.\w+:\d+:\d+).*\[CWE-(.*?)\].*')
    for line in f.readlines():
        mo1 = regex.findall(line)
        if (not mo1):
            continue
        print(mo1)
        mo2 = regex.match(line)
        if (not mo2):
            continue
        print(mo2.group())
