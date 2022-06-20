import sys
import re
import csv
import os
import xml.etree.ElementTree as ET
import numpy as np

####################################################
# USE:
# python3 data-extractor.py dataset-dir TOKEN_WINDOW
####################################################

assert(len(sys.argv) == 3)

###############################
# TOKEN_WINDOW - hyperparameter
###############################

TOKEN_WINDOW = int(sys.argv[2])

dataset_dir = sys.argv[1]

log_metrics = dataset_dir + "/metrics.csv"
log_tokens = dataset_dir + "/tokens.txt"
log_analyzer_gcc = dataset_dir + "/analyzer-gcc.txt"
log_analyzer_cppcheck = dataset_dir + "/analyzer-cppcheck.xml"

#################################
# parse gcc-10 -fanalyzer results
#################################

result_analyzer_gcc = []
with open(log_analyzer_gcc, "r") as f:
    regex = re.compile(r'(.*.[a-z]+):(\d+):(?:.*)\[CWE-(.*?)\]')
    regex_func = re.compile(r'.*In function ‘(.*?)’:$\n')

    cur_cwe = 0
    cur_line = 0
    cur_file = ''
    cur_func = ''

    for line in f.readlines():
        m = regex.findall(line)
        if (m):
            cur_file = m[0][0]
            cur_line = int(m[0][1])
            cur_cwe = int(m[0][2])

            result_analyzer_gcc.append(
                [cur_file, cur_line, cur_cwe])

        m = regex_func.findall(line)
        if (m):
            cur_func = m[0]

########################
# parse cppcheck results
########################

result_analyzer_cppcheck = []
tree = ET.parse(log_analyzer_cppcheck)
root = tree.getroot()
for item in root.findall('./errors/error'):
    cur_cwe = int(item.get('cwe'))
    for i in item.findall('./location'):
        cur_file = file = os.path.basename(i.get('file'))
        cur_line = int(i.get('line'))
        result_analyzer_cppcheck.append([cur_file, cur_line, cur_cwe])

########################
# merge analyzer results
########################

result_analyzer_gcc = np.array(result_analyzer_gcc)
result_analyzer_cppcheck = np.array(result_analyzer_cppcheck)

result_analyzer = result_analyzer_gcc

if (result_analyzer.size == 0):
    result_analyzer = result_analyzer_cppcheck
else:
    for r_cppcheck in result_analyzer_cppcheck:
        present = False
        for r in result_analyzer:
            if (np.array_equal(r_cppcheck, r)):
                present = True
                break
        if (not present):
            result_analyzer = np.vstack((result_analyzer, r_cppcheck))

if (result_analyzer.size == 0):
    # print("NO ERRORS DETECTED IN")
    exit(0)

################
# process tokens
################


class Func:
    def __init__(self, file, name, line_start, line_finish, idx):
        self.file = file
        self.name = name
        self.line_start = line_start
        self.line_finish = line_finish
        self.idx = idx


result_tokens = []
functions = []
with open(log_tokens, "r") as f:
    regex_file = re.compile(r'Start lexing translation unit: (.*)')
    regex_func = re.compile(r'\[fn:(.*)@(\d+)-(\d+)')
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

                line_start = int(m[0][1])
                line_finish = int(m[0][2])
                functions.append(Func(cur_file, func_name, line_start,
                                      line_finish, len(result_tokens)))
            continue

        if (line.startswith('  (')):
            m = regex_token.findall(line)
            if (m):
                for i in m:
                    result_tokens.append([i[1], i[4], i[5]])
            continue

print(result_analyzer)

#####################################
# obtaining token window for each cwe
#####################################


def is_valid_token(token):
    if ((token >= 4 and token <= 22) or
            token == 60 or token == 61 or token >= 76):
        return True
    return False


result_analyzer_extended = []

for r in range(len(result_analyzer)):
    cwe_file = result_analyzer[r][0]
    cwe_line = int(result_analyzer[r][1])
    f_name = ''
    token_idx = 0
    token_idx_start = 0
    token_idx_end = 0
    line_finish = 0

    for f in functions:
        line_in_bounds = ((f.line_start <= cwe_line) and (
            f.line_finish >= cwe_line))
        if (f.file == cwe_file and line_in_bounds):
            f_name = f.name
            token_idx_start = f.idx
            line_finish = f.line_finish
            break
    assert(f_name)

    ###########################################
    # search for idx of first token in cwe line
    ###########################################

    for i in range(token_idx_start, len(result_tokens)):
        idx = int(result_tokens[i][0])
        if (cwe_line == idx):
            token_idx = i
        if (line_finish == idx):
            token_idx_end = i
        if (token_idx and token_idx_end):
            break

    assert(token_idx and token_idx_end)

    print(token_idx)
    print(token_idx_end)

    ##############################################################################
    # get context - [cwe_line_start - TOKEN_WINDOW, cwe_line_start + TOKEN_WINDOW]
    ##############################################################################

    context = np.zeros(2 * TOKEN_WINDOW)

    ######################
    # forward fill context
    ######################

    idx = token_idx
    for i in range(TOKEN_WINDOW, 2 * TOKEN_WINDOW):
        # while (idx < len(result_tokens)):  # emit full context
        while (idx < token_idx_end):   # only emit function's tokens
            cur_token = int(result_tokens[idx][1])
            if (is_valid_token(cur_token)):
                context[i] = cur_token
                idx += 1
                break
            idx += 1

    #######################
    # backward fill context
    #######################

    idx = token_idx
    for i in reversed(range(TOKEN_WINDOW)):
        # while (idx > 0):  # emit full context
        while (idx > token_idx_start):
            cur_token = int(result_tokens[idx][1])
            if (is_valid_token(cur_token)):
                context[i] = cur_token
                idx -= 1
                break
            idx -= 1

    print(context)

    result_analyzer_extended.append(
        np.append(np.append(f_name, result_analyzer[r]), context))


####################
# parse code metrics
####################

result_metrics = []
with open(log_metrics, "r") as f:
    csvreader = csv.reader(f)
    rows = []
    for row in csvreader:
        rows.append(row)
    for r in rows:
        file = os.path.basename(r[0])
        r[0] = file
        func = r[1]
        result_metrics.append(r)

###############
# merge dataset
###############

dataset = []
for r in result_analyzer_extended:
    for m in result_metrics:
        if (r[1] == m[0] and r[0] == m[1]):
            dataset.append(np.append(r[3:], m[2:]))

with open(dataset_dir + "/data.csv", "w") as d:
    csvwriter = csv.writer(d)
    csvwriter.writerows(dataset)
