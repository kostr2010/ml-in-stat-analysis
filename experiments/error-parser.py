# import re
# import sys

# # use: <python interpreter> error-parser.py <file in> <file out>
# assert(len(sys.argv) == 3)

# file_in = sys.argv[1]
# file_out = sys.argv[2]

# result = []

# with open(file_in) as f:
#     regex = re.compile(r'(.*\w+.\w+:\d+:\d+)(?:.*)\[CWE-(.*?)\]')
#     regex_func = re.compile(r'.*In function ‘(.*?)’:$\n')

#     cur_path = ''
#     cur_func = ''
#     cur_cwe = 0

#     for line in f.readlines():
#         m = regex.findall(line)
#         if (m):
#             cur_path = m[0][0]
#             cur_cwe = m[0][1]
#             result.append([cur_path, cur_func, cur_cwe])

#         m = regex_func.findall(line)
#         if (m):
#             cur_func = m[0]


# print(result)
import sys
import re
import csv
import os
import xml.etree.ElementTree as ET

# process tokens


class Token:
    def __init__(self, name, kind, line, flags):
        self.name = name
        self.kind = kind
        self.line = line
        self.flags = flags


class Func:
    def __init__(self, file, name, line_start, line_finish, idx):
        self.file = file
        self.name = name
        self.line_start = line_start
        self.line_finish = line_finish
        self.idx = idx


result_tokens = []
functions = []
with open("/home/kostr2010/cod/ml-in-stat-analysis/build/src/dataset/CWE415_Double_Free__malloc_free_struct_61/tokens.txt", "r") as f:
    regex_file = re.compile(r'Start lexing translation unit: (.*)')
    regex_func = re.compile(r'\[fn:(.*(good|bad).*)@(\d+)-(\d+)')
    regex_token = re.compile(r'\((\w+),(\d+),(\d+),(\d+),(\d+),(\d+)\)')
    cur_file = ''
    for line in f.readlines():
        if (line.startswith('S')):
            m = regex_file.findall(line)
            cur_file = os.path.basename(m[0])

        if (line.startswith('[fn:')):
            m = regex_func.findall(line)
            if (m):
                func_name = m[0][0]

                present = False
                for f in functions:
                    if (f.name == func_name and f.file == cur_file):
                        present = True
                        break
                if present:
                    continue

                is_good = m[0][1] == 'good'
                line_start = int(m[0][2])
                line_finish = int(m[0][3])
                functions.append(Func(cur_file, func_name, line_start,
                                      line_finish, len(result_tokens)))
            continue

        if (line.startswith('  (')):
            m = regex_token.findall(line)
            if (m):
                for i in m:
                    result_tokens.append([i[1], i[4], i[5]])
            continue

for f in functions:
    print(f.file + ":" + f.name)
    print(result_tokens[f.idx])
    print(result_tokens[f.idx + 1])
