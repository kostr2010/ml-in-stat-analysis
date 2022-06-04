import sys
import re
import csv
import os
import xml.etree.ElementTree as ET
import numpy as np

###############################################################
# USE:
# python3 data-extractor.py dataset-dir juliet-dir TOKEN_WINDOW
###############################################################

assert(len(sys.argv) == 4)

###############################
# TOKEN_WINDOW - hyperparameter
###############################

TOKEN_WINDOW = int(sys.argv[3])

dataset_dir = sys.argv[1]
juliet_dir = sys.argv[2]

log_metrics = dataset_dir + "/metrics.csv"
log_tokens = dataset_dir + "/tokens.txt"
log_analyzer_gcc_good = dataset_dir + "/analyzer-gcc-good.txt"
log_analyzer_gcc_bad = dataset_dir + "/analyzer-gcc-bad.txt"
log_analyzer_cppcheck_good = dataset_dir + "/analyzer-cppcheck-good.xml"
log_analyzer_cppcheck_bad = dataset_dir + "/analyzer-cppcheck-bad.xml"

reg = re.compile(r'CWE(\d+)')
true_cwe = int(reg.findall(dataset_dir)[0])

#################################
# parse gcc-10 -fanalyzer results
#################################

result_analyzer_gcc = []
with open(log_analyzer_gcc_bad, "r") as f:
    regex = re.compile(
        r'/(CWE(\d+)_[\w,_]*__.*_\d\d.*.c):(\d+):\d+(?:.*)\[CWE-(.*?)\]')
    regex_func = re.compile(r'.*In function ‘(.*?)’:$\n')

    cur_file = ''
    cur_line = ''
    cur_func = ''
    cur_cwe = 0

    for line in f.readlines():
        m = regex.findall(line)
        if (m):
            cur_file = m[0][0]
            cur_line = int(m[0][2])
            cur_cwe = int(m[0][3])

            if (cur_cwe != true_cwe):
                continue

            result_analyzer_gcc.append(
                [cur_file, cur_line, True, cur_cwe])

        m = regex_func.findall(line)
        if (m):
            cur_func = m[0]

with open(log_analyzer_gcc_good, "r") as f:
    regex = re.compile(
        r'/(CWE(\d+)_[\w,_]*__.*_\d\d.*.c):(\d+):\d+(?:.*)\[CWE-(.*?)\]')
    regex_func = re.compile(r'.*In function ‘(.*?)’:$\n')

    cur_file = ''
    cur_line = ''
    cur_func = ''
    cur_cwe = 0

    for line in f.readlines():
        m = regex.findall(line)
        if (m):
            cur_file = m[0][0]
            cur_line = int(m[0][2])
            cur_cwe = int(m[0][3])

            if (cur_cwe != true_cwe):
                continue

            print("GOT FP (GCC)! " + cur_file +
                  ":" + str(cur_line) + " CWE-" + str(cur_cwe))

            result_analyzer_gcc.append(
                [cur_file, cur_line, False, cur_cwe])

        m = regex_func.findall(line)
        if (m):
            cur_func = m[0]

########################
# parse cppcheck results
########################

result_analyzer_cppcheck = []
tree = ET.parse(log_analyzer_cppcheck_bad)
root = tree.getroot()
for item in root.findall('./errors/error'):
    cur_cwe = int(item.get('cwe'))
    if (cur_cwe != true_cwe):
        continue
    for i in item.findall('./location'):
        cur_file = file = os.path.basename(i.get('file'))
        cur_line = int(i.get('line'))
        result_analyzer_cppcheck.append([cur_file, cur_line, True, cur_cwe])

tree = ET.parse(log_analyzer_cppcheck_good)
root = tree.getroot()
for item in root.findall('./errors/error'):
    cur_cwe = int(item.get('cwe'))
    if (cur_cwe != true_cwe):
        continue
    for i in item.findall('./location'):
        cur_file = file = os.path.basename(i.get('file'))
        cur_line = int(i.get('line'))
        print("GOT FP (CPPCHECK)! " + cur_file +
              ":" + str(cur_line) + " CWE-" + str(cur_cwe))
        result_analyzer_cppcheck.append(
            [cur_file, cur_line, False, cur_cwe])

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

#####################################
# obtaining token window for each cwe
#####################################

result_analyzer_extended = []

for r in range(len(result_analyzer)):
    cwe_file = result_analyzer[r][0]
    cwe_line = int(result_analyzer[r][1])
    f_name = ''
    token_idx = 0

    for f in functions:
        line_in_bounds = ((f.line_start <= cwe_line) and (
            f.line_finish >= cwe_line))
        if (f.file == cwe_file and line_in_bounds):
            f_name = f.name
            token_idx = f.idx
            break
    assert(f_name)

    ###########################################
    # search for idx of first token in cwe line
    ###########################################

    for i in range(token_idx, len(result_tokens)):
        if (cwe_line == int(result_tokens[i][0])):
            token_idx = i
            break

    ##############################################################################
    # get context - [cwe_line_start - TOKEN_WINDOW, cwe_line_start + TOKEN_WINDOW]
    ##############################################################################

    context = np.zeros(2 * TOKEN_WINDOW)

    COMMENT_TOKEN_ID = 4
    context.fill(COMMENT_TOKEN_ID)

    ######################
    # forward fill context
    ######################

    for i in range(TOKEN_WINDOW, 2 * TOKEN_WINDOW):
        idx = i - TOKEN_WINDOW + token_idx
        if (idx >= len(result_tokens)):
            break
        context[i] = result_tokens[idx][1]

    #######################
    # backward fill context
    #######################

    for i in reversed(range(TOKEN_WINDOW)):
        idx = i - TOKEN_WINDOW + token_idx
        if (idx < 0):
            break
        context[i] = result_tokens[idx][1]

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
        regex = re.compile(r'(CWE\d\d\d_[\w, _]*__.*_\d\d.*.c)')
        m = regex.findall(file)
        if (m):
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
